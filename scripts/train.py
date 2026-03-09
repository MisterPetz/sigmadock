import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml
from pytorch_lightning import Trainer, callbacks, seed_everything
from pytorch_lightning.loggers import WandbLogger
from rdkit import RDLogger

from sigmadock.config import (
    RunConfig,
    StructuralConfig,
    get_exp_dir,
    get_exp_dir_from_ckpt,
    get_experiment_config,
    parse_args,
    update_config_from_args,
)
from sigmadock.core.callbacks import EMAWithRampup, FullNaNCheckCallback, SamplerDebugCallback
from sigmadock.data import SigmaDataModule
from sigmadock.datafronts import MetaFront
from sigmadock.diff.denoiser import SigmaDockDenoiser
from sigmadock.net.model import EquiformerV2
from sigmadock.oracle import HPARAMS
from sigmadock.torch_utils.utils import extract_init_kwargs
from sigmadock.trainer import SigmaLightningModule
from sigmadock.utils import get_git_commit_hash

# --- Suppress RDKit logs ---
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def main() -> None:  # noqa: C901
    cli = parse_args()
    cfg = RunConfig()

    # Load YAML config if provided (CLI overrides config file)
    if getattr(cli, "config", None) is not None:
        with open(cli.config) as f:
            raw = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(RunConfig)}
        yaml_overrides = {}
        for k, v in raw.items():
            if k not in valid or v is None:
                continue
            if k in ("data_dir", "exp_dir") and isinstance(v, str):
                v = Path(v)
            elif k == "betas" and isinstance(v, list):
                v = tuple(v)
            yaml_overrides[k] = v
        cfg = replace(cfg, **yaml_overrides)

    # CLI overrides (including data_dir, num_workers when using --config)
    overrides = {k: v for k, v in vars(cli).items() if v is not None and k != "config"}
    args = replace(cfg, **overrides)

    assert args.data_dir.exists(), f"Data directory {args.data_dir} does not exist."
    print(f"Using data directory: {args.data_dir}")
    print(f"Using training experiments: {args.train_exps}")
    print(f"Using validation experiments: {args.val_exps}")

    # Set kernel CUDA precision
    if torch.cuda.is_available():
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision(args.cuda_precision)
    else:
        print("CUDA not available; running on CPU.")

    # Check for the presence of a checkpoint file and resume if required
    checkpoint_dir = args.exp_dir / "checkpoints"
    last_ckpt = checkpoint_dir / "last.ckpt"
    best_ckpts = sorted(checkpoint_dir.glob("best-checkpoint-*.ckpt"))

    if args.resume_from_checkpoint:
        if isinstance(args.resume_from_checkpoint, str) and Path(args.resume_from_checkpoint).exists():
            # If a valid checkpoint path is provided, resume from it
            ckpt_path = args.resume_from_checkpoint
            EXP_DIR = get_exp_dir_from_ckpt(ckpt_path)  # Use the checkpoint directory for EXP_DIR
            print(f"Resuming from provided checkpoint: {ckpt_path}")
        elif last_ckpt.exists():
            # If no specific checkpoint was provided, but "last.ckpt" exists, resume from it
            ckpt_path = str(last_ckpt)
            EXP_DIR = get_exp_dir_from_ckpt(ckpt_path)  # Use checkpoint directory for EXP_DIR
            print(f"Resuming from last checkpoint: {ckpt_path}")
        elif best_ckpts:
            # If no specific checkpoint was provided, but best checkpoints exist, resume from the latest one
            ckpt_path = str(best_ckpts[-1])
            EXP_DIR = get_exp_dir_from_ckpt(ckpt_path)  # Use checkpoint directory for EXP_DIR
            print(f"Resuming from best checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found, starting fresh training.")
            EXP_DIR = get_exp_dir(args)
    else:
        print("Starting fresh training run.")
        EXP_DIR = get_exp_dir(args)

    # Setup experiment paths
    wandb_dir = EXP_DIR / "wandb_logs"
    ckpt_dir = EXP_DIR / "checkpoints"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume W&B run if specified
    wandb_wandb = wandb_dir / "wandb"
    resume_id = None
    if args.resume_from_checkpoint:
        latest = wandb_wandb / "latest-run"
        if latest.exists():
            # Resolve the symlink (e.g. "run-20250620_125605-ptzprs00")
            target = os.readlink(latest)
            # Extract the run-id (the part after the last dash)
            resume_id = Path(target).name.split("-")[-1]
            print(f"Auto-detected W&B run ID: {resume_id}")

    # Seed everything for reproducibility
    seed_everything(seed=args.seed, workers=True)
    if args.debug:
        print(f"[WARN] Debug mode enabled. Seed: {args.seed}")

    # Extract configs from args
    train_cfgs = [get_experiment_config(name, root_dir=args.data_dir) for name in args.train_exps]
    val_cfgs = [get_experiment_config(name, root_dir=args.data_dir) for name in args.val_exps]
    test_cfgs = [get_experiment_config(name, root_dir=args.data_dir) for name in args.test_exps]

    # Create datafronts
    train_datafront = MetaFront(train_cfgs)
    val_datafront = MetaFront(val_cfgs)
    test_datafront = MetaFront(test_cfgs)
    print(f"Training datafront: {train_datafront}")
    print(f"Validation datafront: {val_datafront}")
    print(f"Test datafront: {test_datafront}")

    # Update structural config from args
    structural_config = StructuralConfig()
    structural_config = update_config_from_args(structural_config, args)

    # Generate DataModule
    datamodule = SigmaDataModule(
        seed=args.seed,
        train_datafront=train_datafront,
        val_datafront=val_datafront,
        test_datafront=test_datafront,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        # Cycle caching. Speedup = cache_cycles / cache_factor.
        cache_factor=args.cache_factor,
        cache_cycles=args.cache_cycles,
        dataset_augmentation_factor=args.dataset_augmentation_factor,
        # Validation Cycles
        val_cycles=args.val_cycles,
        # Cheminformatics
        **structural_config.__dict__,
    )

    equimodel = EquiformerV2(
        use_esm_embeddings=args.use_esm_embeddings,
        # Base Layers
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        atom_feature_dims=args.atom_feature_dims,
        average_degrees=args.average_degrees,
        edge_feature_dims=args.edge_feature_dims,
        lmax_list=args.l_max_list,
        mmax_list=args.m_max_list,
        # Interactions
        protein_ligand_interactions=args.include_protein_ligand_interactions,
        ligand_ligand_interactions=args.include_fragment_fragment_interactions,
        # Dimensions
        sphere_channels=args.sphere_channels,
        edge_channels=args.edge_channels,
        # Hidden layers
        attn_hidden_channels=args.attn_hidden_channels,
        attn_alpha_channels=args.attn_alpha_channels,
        attn_value_channels=args.attn_value_channels,
        ffn_hidden_channels=args.ffn_hidden_channels,
        # Distance & Time Dimensions
        t_emb_dim=args.t_emb_dim,
        t_emb_type=args.t_emb_type,
        t_emb_scale=args.t_emb_scale,
        distance_expansion_dim=args.distance_expansion_dim,
        smearing_type=args.smearing_type,
        radial_cutoff_function=args.radial_type,
        rel_distance=args.rel_distance,
        # Dropout
        alpha_drop=args.attention_dropout,
        drop_path_rate=args.edge_dropout,
        # Zero weight init at force block output
        zero_init_last=args.zero_init_last,
        share_edge_mlp=False,
    )
    # Init denoiser
    denoiser = SigmaDockDenoiser(
        equimodel,
        cache_path=EXP_DIR.parent / "cache",
        # Cutoffs for local dynamic edges
        cutoff_complex_interactions=HPARAMS.get_edge_spec("inter_complex").r_max
        if args.include_protein_ligand_interactions
        else -1,
        cutoff_fragment_interactions=HPARAMS.get_edge_spec("inter_fragments").r_max
        if args.include_fragment_fragment_interactions
        else -1,
        cutoff_complex_virtual=HPARAMS.get_edge_spec("complex_lv2pv").r_max,
        **args.__dict__,
    )

    # Get configs for loading and saving
    denoiser_cfg: dict[str, Any] = extract_init_kwargs(denoiser, exclude=["model"])
    equiformer_cfg: dict[str, Any] = extract_init_kwargs(equimodel)

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else (args.max_epochs * len(train_datafront) // (args.batch_size * args.world_size))
    )
    max_epochs = (
        args.max_epochs
        if args.max_epochs is not None
        else (args.max_steps * (args.batch_size * args.world_size) // (len(train_datafront)))
    )
    # Load lightning model with denoiser and equiformer
    lightning_model = SigmaLightningModule(
        denoiser=denoiser,
        denoiser_config=denoiser_cfg,
        equiformer_config=equiformer_cfg,
        # Losses
        fragment_scaling=args.fragment_scaling,
        trans_score_weight=args.trans_score_weight,
        rot_score_weight=args.rot_score_weight,
        trans_data_weight=args.trans_data_weight,
        rot_data_weight=args.rot_data_weight,
        # Training
        max_steps=max_steps,
        num_warmup_steps=args.lr_warmup_frac,
        # Optimizers
        min_lr_start=args.min_lr_start,
        cycle_warmup_frac=args.cycle_warmup_frac,
        num_lr_cycles=args.num_lr_cycles,
        init_lr_start=args.init_lr_start,
        min_lr_end=args.min_lr_end,
        max_lr_start=args.max_lr_start,
        max_lr_end=args.max_lr_end,
        weight_decay=args.weight_decay,
        optimizer_eps=args.optimizer_eps,
        betas=args.betas,
        grad_clip=args.grad_clip,
        compile=args.compile,
    )

    # Logging & Checkpointing if not in debug mode
    if not args.offline_run:
        wandb.login()
    run_name = "SigmaDock-Trial" if args.debug else f"{args.experiment}_{args.seed}"

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    # Add git commit hash to config if available
    git_commit = get_git_commit_hash()
    if git_commit:
        cfg["git_commit"] = git_commit
        print(f"Logging with Git commit: {git_commit}")

    wandb_logger = WandbLogger(
        entity="sigma-dock",
        project="SigmaDock",
        name=run_name,
        save_dir=wandb_dir,
        id=resume_id,
        resume="allow" if resume_id else None,
        log_model=(not args.offline_run) and args.log_model,
        offline=args.offline_run,
        config=cfg,
    )

    # Save configs
    with open(EXP_DIR / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)

    # This will call wandb.init() if needed
    run: wandb = wandb_logger.experiment
    run.define_metric(args.monitor_metric, summary="min")
    run.log_code(
        ".",
        include_fn=lambda path: path.endswith(".py"),
        exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("spiral/"),
    )
    all_callbacks = []
    if args.debug:
        all_callbacks.append(SamplerDebugCallback(num_indices=5))
        all_callbacks.append(FullNaNCheckCallback())
        print("[INFO] Sampler Debug Callback and Full NaN Check Callback enabled.")

    if args.early_stopping_patience > 0:
        all_callbacks.append(
            callbacks.EarlyStopping(
                monitor=args.monitor_metric,
                patience=args.early_stopping_patience
                if isinstance(args.early_stopping_patience, int)
                else max_steps * args.early_stopping_patience,
                mode="min",
            )
        )
    else:
        print("[INFO] Early stopping is disabled.")
    all_callbacks += [
        callbacks.ModelCheckpoint(
            save_top_k=3,
            monitor=args.monitor_metric,
            mode="min",
            dirpath=ckpt_dir,
            filename="checkpoint-{step:02d}-{val_loss:.4f}",
            save_weights_only=False,
            save_last=True,
        ),
        callbacks.LearningRateMonitor(logging_interval="step"),
        callbacks.ModelSummary(max_depth=4),
    ]

    # EMA Callback
    if args.use_ema:
        ema_callback = EMAWithRampup(
            ema_halflife_kpoints=args.ema_halflife * len(train_datafront) // 1000,
            ema_rampup_ratio=args.ema_rampup_ratio,
            update_every=2,
            sync_every=8,
            cold_steps=0,
            batch_size=args.batch_size * args.world_size,
            use_ema_for_val=True,
        )
        all_callbacks.append(ema_callback)

    trainer = Trainer(
        precision=args.precision,
        strategy=args.strategy,
        accelerator="cpu" if torch.backends.mps.is_available() else args.accelerator,
        devices=args.devices,
        logger=wandb_logger,
        deterministic=args.deterministic,
        callbacks=all_callbacks,
        max_epochs=max_epochs,
        log_every_n_steps=min(100, len(train_datafront) // (args.batch_size * args.world_size)),
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        profiler="simple" if args.debug else None,
        detect_anomaly=args.debug,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accum_grad_batches,
        use_distributed_sampler=True,
    )
    # Add cfg to trainer
    wandb_logger.log_hyperparams(cfg)

    # Safely check and call trainer.fit()
    try:
        if args.resume_from_checkpoint and "ckpt_path" in locals():
            trainer.fit(lightning_model, datamodule=datamodule, ckpt_path=ckpt_path)
        else:
            trainer.fit(lightning_model, datamodule=datamodule)
    finally:
        wandb.finish()  # Ensures wandb finishes even if an error occurs


if __name__ == "__main__":
    main()
