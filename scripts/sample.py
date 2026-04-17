import contextlib
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader

from sigmadock.chem.postprocessor import compute_gnina_score
from sigmadock.chem.statistics import compact_posebusting
from sigmadock.core.data import SampleCycleWrapper
from sigmadock.data import SigmaDataset
from sigmadock.datafronts import MetaFront
from sigmadock.diff.denoiser import SigmaDockDenoiser
from sigmadock.diff.sampling import sampler
from sigmadock.oracle import HPARAMS
from sigmadock.sampling_setup import (
    build_sampling_datafront,
    experiment_name_is_set,
    prepare_sampling_cfg,
    resolve_sampling_data_dir,
    sampling_results_exp_name,
)
from sigmadock.trainer import SigmaLightningModule
from sigmadock.utils import load_from_scratch

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # go up from scripts/


def _load_pdbs_from_txt(txt_path: Path) -> list[str]:
    """Load PDB IDs from a text file."""
    with open(txt_path) as f:
        pdb_ids = [line.strip() for line in f if line.strip()]
    return pdb_ids


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class SamplingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        predict_dataset: SigmaDataset,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.predict_dataset = predict_dataset

    def setup(self, stage: str) -> None:
        if stage == "predict":
            pass

    def predict_dataloader(self) -> GeometricDataLoader:
        return GeometricDataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
        )


class SamplingModule(pl.LightningModule):
    def __init__(
        self,
        denoiser: SigmaDockDenoiser,
        sampler: Callable,
        seeds: list[int],
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.sampler = sampler
        self.seeds = seeds
        self.cfg = cfg
        
        if self.get_out_dir().exists():
            raise FileExistsError(f"Output directory {self.get_out_dir()} already exists, refusing to overwrite.")

    def configure_optimizers(self) -> None:
        # Not used for sampling
        return None

    def on_predict_start(self) -> None:
        self.results: dict[str, list[dict[str, Any]]] = {}
        # Ensure denoiser is on the module device
        print(f"Setting SO(3) RMS scaling to device to {self.device}.")
        self.denoiser.diffuser._so3_diffuser.set_device(self.device)

    @torch.no_grad()
    def predict_step(self, batch: Batch, batch_idx: int) -> dict[str, Any]:  # noqa: C901
        _, pos, _ = self.sampler(batch=batch)
        is_lig = (batch.frag_idx_map != -1) & (batch.node_entity <= 1)
        is_lig_pos = (batch.node_entity <= 1)[batch.frag_idx_map != -1] 
        per_lig_pocket_com = batch.pocket_com.repeat_interleave(torch.bincount(batch.batch[is_lig]), dim=0)

        # Node-level arrays (device) — we'll slice these per-molecule
        batch_idx_nodes = batch.batch[is_lig]  # [num_ligand_nodes] (device)
        x0_all = batch.pos_0[is_lig] * HPARAMS.general.dimensional_scale + per_lig_pocket_com  # [num_lig_nodes,3]
        x0_hat_all = pos[-1][is_lig_pos] * HPARAMS.general.dimensional_scale + per_lig_pocket_com  # [num_lig_nodes,3]
        # traj_all = pos[is_lig_pos[None]]  # [T, num_lig_nodes, 3]
        traj_all = torch.stack([p[is_lig_pos] for p in pos])  # [T, num_lig_nodes, 3]
        com_all = batch.pocket_com  # [num_molecules, 3] (device)

        # Mol-level metadata (may be tensor or python list)
        mol_ids_all = batch.mol_info["mol_id"]
        lig_ref_all = batch.mol_info["original"]
        prot_ref_all = batch.mol_info["pocket"]
        ligand_path_all = batch.mol_info["ligand_path"]
        pdb_path_all = batch.mol_info["pdb_path"]
        seeds_all = batch.seed
        ref_for_pocket_all = batch.mol_info.get("reference")

        # Get PDB-LIG code from pdb_path_all
        pdb_lig_codes: list[str] = ["_".join(Path(p).stem.split("_")[:2]) for p in pdb_path_all]
        
        # Build the original batch-level out (keeps backwards compatibility)
        out = {
            "com": com_all.cpu(),
            "batch_idx": batch_idx_nodes.cpu(),
            "x0": x0_all.cpu(),
            "x0_hat": x0_hat_all.cpu(),
            "trajectory": traj_all.cpu(),
            "lig_ref": lig_ref_all,
            "prot_ref": prot_ref_all,
            "mol_id": (mol_ids_all.cpu() if torch.is_tensor(mol_ids_all) else mol_ids_all),
            "ligand_path": (ligand_path_all.cpu() if torch.is_tensor(ligand_path_all) else ligand_path_all),
            "pdb_path": (pdb_path_all.cpu() if torch.is_tensor(pdb_path_all) else pdb_path_all),
            "seed": (seeds_all.cpu() if torch.is_tensor(seeds_all) else seeds_all),
            "pdb_lig_codes": pdb_lig_codes,
        }

        try:
            num_mols = com_all.shape[0]
        except Exception:
            num_mols = len(mol_ids_all) if (not torch.is_tensor(mol_ids_all)) else int(mol_ids_all.numel())

        # --- Split into per-molecule entries and append to self.results ---
        # unique molecule indices present among ligand nodes (sorted)
        unique_mols = torch.unique(batch_idx_nodes, sorted=True)

        # helper to read mol_id / seed for molecule index i
        def _get_mol_id(i: int) -> Any:
            if torch.is_tensor(mol_ids_all):
                if mol_ids_all.ndim == 0:
                    return mol_ids_all.item()
                return mol_ids_all[i].item() if mol_ids_all[i].numel() == 1 else mol_ids_all[i]
            else:
                try:
                    return mol_ids_all[i]
                except Exception:
                    return mol_ids_all

        def _get_seed(i: int) -> int:
            if torch.is_tensor(seeds_all):
                if seeds_all.ndim == 0:
                    return int(seeds_all.item())
                # if per-molecule seeds provided
                if seeds_all.numel() == num_mols:
                    return int(seeds_all[i].item())
                # otherwise fallback to scalar seed
                return int(seeds_all.item())
            else:
                try:
                    return seeds_all[i]
                except Exception:
                    return seeds_all

        # For each molecule index present in this batch, slice node-level tensors and store a per-mol dict
        for mi in unique_mols:
            mi = int(mi.item())  # molecule index within this batch
            # indices of ligand nodes that belong to molecule `mi`
            mask = batch_idx_nodes == mi
            node_idxs = torch.nonzero(mask, as_tuple=True)[0]  # 1D long tensor (may be empty)

            # slice node-level arrays (if no nodes, produce empty tensors with correct trailing dim)
            if node_idxs.numel() > 0:
                x0_i = x0_all[node_idxs]  # [num_nodes_i, 3] (device)
                x0_hat_i = x0_hat_all[node_idxs]  # [num_nodes_i, 3] (device)
                traj_i = traj_all[:, node_idxs, :]  # [T, num_nodes_i, 3] (device)
            else:
                # empty fallback shapes
                x0_i = torch.empty((0, x0_all.shape[-1]), device=x0_all.device)
                x0_hat_i = torch.empty((0, x0_hat_all.shape[-1]), device=x0_hat_all.device)
                if traj_all is not None and traj_all.ndim >= 3:
                    T = traj_all.shape[0]
                    traj_i = torch.empty((T, 0, traj_all.shape[-1]), device=x0_all.device)
                else:
                    traj_i = torch.empty((0, 0, 3), device=x0_all.device)

            # per-molecule metadata
            com_i = com_all[mi] if torch.is_tensor(com_all) else com_all[mi]  # noqa: RUF034
            mol_id_i = _get_mol_id(mi)
            seed_i = _get_seed(mi)
            lig_ref_i = lig_ref_all[mi] if (hasattr(lig_ref_all, "__getitem__")) else lig_ref_all
            prot_ref_i = prot_ref_all[mi] if (hasattr(prot_ref_all, "__getitem__")) else prot_ref_all
            ligand_path_i = ligand_path_all[mi] if (hasattr(ligand_path_all, "__getitem__")) else ligand_path_all
            pdb_path_i = pdb_path_all[mi] if (hasattr(pdb_path_all, "__getitem__")) else pdb_path_all
            pdb_lig_code_i = pdb_lig_codes[mi] if (hasattr(pdb_lig_codes, "__getitem__")) else pdb_lig_codes
            # One key per dataset item: same protein + different query SDFs (e.g. CSV datafront) must not
            # collapse under pdb_lig_code alone, or PoseBusters sees multiple list entries per key.
            lig_stem = Path(str(ligand_path_i)).stem
            results_key = f"{pdb_lig_code_i}::{lig_stem}"

            # move to CPU and convert small scalars to python types
            com_i_cpu = com_i.detach().cpu() if torch.is_tensor(com_i) else com_i
            x0_i_cpu = x0_i.detach().cpu()
            x0_hat_i_cpu = x0_hat_i.detach().cpu()
            traj_i_cpu = traj_i.detach().cpu() if torch.is_tensor(traj_i) else torch.as_tensor(traj_i)

            if torch.is_tensor(mol_id_i) and mol_id_i.numel() == 1:
                with contextlib.suppress(Exception):
                    mol_id_i = mol_id_i.item()
            if torch.is_tensor(seed_i) and getattr(seed_i, "numel", lambda: 1)() == 1:
                seed_i = int(seed_i.item())

            crossdocking_i = False
            if ref_for_pocket_all is not None:
                try:
                    ref_p = ref_for_pocket_all[mi] if hasattr(ref_for_pocket_all, "__getitem__") else ref_for_pocket_all
                    crossdocking_i = ref_p is not None
                except (TypeError, IndexError, KeyError):
                    crossdocking_i = False

            per_mol_out = {
                "seed": seed_i,
                "com": com_i_cpu,  # [3] tensor
                "x0": x0_i_cpu,  # [num_nodes_i, 3] tensor
                "x0_hat": x0_hat_i_cpu,  # [num_nodes_i, 3] tensor
                "trajectory": traj_i_cpu,  # [T, num_nodes_i, 3] tensor
                "lig_ref": lig_ref_i,
                "prot_ref": prot_ref_i,
                "ligand_path": ligand_path_i,
                "pdb_path": pdb_path_i,
                "mol_id": mol_id_i,
                "crossdocking": crossdocking_i,
            }
            # append one dict per molecule (per seed)
            if results_key in self.results:
                self.results[results_key].append(per_mol_out)
            else:
                self.results[results_key] = [per_mol_out]
        return out

    def _gather_python_objects(self, local_obj: Any) -> list:
        """
        Gather arbitrary python objects from all ranks.
        Preferred: use torch.distributed.all_gather_object (clean).
        Fallback: pickle into uint8 tensor, pad to max length and gather with dist.all_gather.
        Returns a flattened list of gathered objects (one list per rank flattened).
        """
        if not (dist.is_available() and dist.is_initialized()):
            # single-process - just return local object wrapped as list for compatibility
            return [local_obj]

        world_size = dist.get_world_size()

        # try the easy direct API first
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_obj)
        return gathered

    def on_predict_epoch_end(self, outputs: Any = None) -> None:
        # gather per-rank dicts (collective must be invoked on all ranks)
        gathered = self._gather_python_objects(self.results)

        # ensure it's a list (one element per rank)
        if not isinstance(gathered, list):
            gathered = [gathered]

        # merge into one dict: global_results[key] -> list of per_mol_out
        global_results = {}
        for rank_dict in gathered:
            if not rank_dict:
                continue
            for key, lst in rank_dict.items():
                # each rank_dict[key] is a list[per_mol_out]
                global_results.setdefault(key, []).extend(lst)

        # only rank 0 writes/does heavy collation
        is_rank0 = True
        if dist.is_available() and dist.is_initialized():
            is_rank0 = dist.get_rank() == 0
        if not is_rank0:
            return
        self.save_results(global_results)

    def get_out_dir(self) -> Path:
        exp_name = sampling_results_exp_name(self.cfg)
        global_seed = int(self.cfg.seed)

        model_id = (
            str(self.cfg.model.model_id) if self.cfg.model.model_id is not None else Path(self.cfg.model.ckpt_dir).stem
        )
        root = PROJECT_ROOT if self.cfg.output_dir is None else Path(self.cfg.output_dir)
        out_dir = root / "results" / exp_name / model_id / f"seed_{global_seed}"
        return out_dir
    
    @rank_zero_only
    def save_results(self, results: dict) -> None:
        # Basic metadata and paths
        out_dir = self.get_out_dir()
        if out_dir.exists():
            print(f"Output directory {out_dir} exists, may contain previous results.")
            # Add a numeric suffix to avoid clobbering
            for suffix in range(1, 1000):
                candidate = out_dir.parent / f"seed_{int(self.cfg.seed)}_{suffix}"
                if not candidate.exists():
                    out_dir = candidate
                    print(f"Using {out_dir} instead.")
                    break
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except Exception as e:
            print(f"Failed to create output directory {out_dir}: {e}")
            # Another process may have created it between exists() and mkdir(); accept that.
            pass

        base_name = "predictions.pt"
        out_path = out_dir / base_name

        meta = {
            "saved_at_utc": ".".join((datetime.utcnow().isoformat() + "Z").split(".")[:-1]),
            "torch_version": torch.__version__,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "world_size": dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1,
            "rank": dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0,
        }

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        try:
            torch.save({"results": results, "meta": meta}, tmp_path, _use_new_zipfile_serialization=True)
            os.replace(tmp_path, out_path)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

        # Save config as YAML beside the .pt file.
        cfg_path = out_path.with_suffix(".yaml")
        OmegaConf.save(config=self.cfg, f=str(cfg_path))

        # Perform PostProcessing here as well if specified - e.g. compute Vina scores
        if self.cfg.postprocessing.scoring is not None:
            assert self.cfg.postprocessing.scoring in ["vinardo", "vina"], (
                f"Unknown scoring method: {self.cfg.postprocessing.scoring}"
            )
            try:
                scores, failed = compute_gnina_score(
                    out_path, scoring=self.cfg.postprocessing.scoring, preprocess=False, no_gpu=True, device=None
                )
                # score_save_dir = out_path.parent / (f"rescoring_{self.cfg.postprocessing.scoring}.pt")
                score_save_dir = out_path.parent / "rescoring.pt"
                torch.save(
                    {"scores": scores, "failed": failed, "score_config": self.cfg.postprocessing}, score_save_dir
                )
            except Exception as e:
                print(f"Failed to compute gnina scores: {e}. Continuing...")
                pass

        if self.cfg.postprocessing.bust_config is not None:
            assert self.cfg.postprocessing.bust_config in ["redock", "redock-fast"], (
                f"Unknown posebusting config: {self.cfg.postprocessing.bust_config}"
            )
            all_rmsds, all_pb_checks, all_pb_dicts = compact_posebusting(
                results, config=self.cfg.postprocessing.bust_config
            )
            pb_save_dir = out_path.parent / "posebusters.pt"
            torch.save(
                {"rmsds": all_rmsds, "pb_checks": all_pb_checks, "pb_dicts": all_pb_dicts},
                pb_save_dir,
            )


@hydra.main(version_base=None, config_path="../conf/", config_name="sampling/base")
def sample(global_cfg: DictConfig) -> None:
    """Run sampling with the given configuration."""
    cfg = prepare_sampling_cfg(global_cfg)

    # Configure Hardware
    torch.set_float32_matmul_precision(cfg.hardware.cuda_precision)

    # Configure seed
    SEED = cfg.seed
    pl.seed_everything(SEED, workers=True)
    # Generate random num_seeds different from generator
    generator = torch.Generator().manual_seed(SEED)
    seeds = torch.randint(0, 1000000, (cfg.num_seeds,), generator=generator).tolist()
    num_seeds: int = len(seeds)
    if num_seeds > 1:
        print(
            "[WARN] num_seeds>1 packs multiple stochastic draws into one job (effective batch_size * num_seeds); "
            "reproducibility is weaker than separate runs."
        )
        print(
            "[WARN] For full reproducibility and top-from-N over independent samples, use num_seeds=1 and "
            "multiple runs with different `seed` (e.g. SLURM job array; see README and slurm/sample.sh)."
        )
    print(f"Using {num_seeds} seeds: {seeds} generated with source seed {SEED}.")

    # Load directories
    CKPT_DIR = Path(cfg.model.ckpt_dir)
    if cfg.model.ckpt_dir is None or str(cfg.model.ckpt_dir).strip() == "":
        raise ValueError("Checkpoint path (model.ckpt_dir) must be specified in the configuration.")

    DATA_DIR = resolve_sampling_data_dir(cfg)
    if not DATA_DIR.exists():
        raise ValueError(f"Data directory does not exist: {DATA_DIR}")

    datafront = build_sampling_datafront(cfg, DATA_DIR)

    if (
        cfg.data.blacklist
        and experiment_name_is_set(cfg.experiments.get("name"))
        and str(cfg.experiments.name).lower() == "posebusters"
        and isinstance(datafront, MetaFront)
    ):
        print("Pruning datafront with (white) blacklist...")
        selected_ids = _load_pdbs_from_txt(Path(cfg.data.blacklist))
        datafront.prune_pairs_with_ids(selected_ids)

    # Create dataset
    dataset = SigmaDataset(
        datafront=datafront,
        # Pocket Graph Definition
        pocket_com_noise=cfg.graph.pocket_com_noise,
        pocket_distance_cutoff=cfg.graph.pocket_distance_cutoff,
        pocket_distance_noise=cfg.graph.pocket_distance_noise,
        prot_coordinate_distance_noise=cfg.graph.pocket_coordinate_jitter,
        use_esm_embeddings=cfg.graph.use_esm_embeddings,
        ignore_triangulation=cfg.graph.ignore_triangulation,
        lig_coordinate_distance_noise=0.0,
        # NOTE: Should get pocket_virtual_cutoff from checkpoint for safety (Hardcoded to init default due to changes in codebase with current .ckpt).  # noqa: E501
        # TODO: Re-incoporate in future versions:
        # pocket_virtual_cutoff=HPARAMS.get_edge_spec("protein_v2v").r_max,
        # Fragmentation
        alignment_tries=cfg.graph.alignment_tries,
        fragmentation_strategy=cfg.graph.fragmentation_strategy,
        ignore_conjugated_torsions=cfg.graph.ignore_conjugated_torsion,
        # Misc
        pb_check=False,
        get_mol_info=True,
        seed=SEED,
        random_rotation=cfg.graph.random_rotation,
        # Sampling-specific
        sample_conformer=cfg.graph.sample_conformer,  # Sample conformers at inference
        skip_bounds_check=True,  # Don't care about torsional bounds and max_weight during sampling at inference.
        # Forced safeguards
        force_retry=True,
    )

    # num_seeds>1: repeat each dataset index num_seeds times (stochastic fragmentation per draw).
    # Prefer num_seeds=1 + job array with different cfg.seed for reproducible independent runs (see README).
    recycle_dataset = (
        SampleCycleWrapper(
            base_dataset=dataset,
            num_samples=num_seeds,
        )
        if num_seeds > 1
        else dataset
    )
    effective_batch_size = int(cfg.data.batch_size) * num_seeds
    print(
        f"Effective batch size (batch_size x num_seeds): {cfg.data.batch_size} x {num_seeds} = {effective_batch_size}"
    )
    if num_seeds > 1:
        print(
            "[WARN] Large effective batch from num_seeds>1 can OOM; consider num_seeds=1 and a job array instead."
        )

    datamodule = SamplingDataModule(
        predict_dataset=recycle_dataset,
        num_workers=int(cfg.data.num_workers),
        batch_size=effective_batch_size,
    )
    datamodule.setup("predict")

    print(f"Dataset size: {len(dataset)}")

    # Load model from checkpoint
    model: SigmaLightningModule = load_from_scratch(
        CKPT_DIR,
        load_ema=cfg.model.use_ema,
        enforced_cfg={
            "cache_path": PROJECT_ROOT / cfg.model.cached_s03_dir,
        }
        if cfg.model.cached_s03_dir is not None
        else None,
        strict=True,
    )
    if cfg.model.use_ema:
        # Use the EMA model for inference
        lightning_model = model.ema_model
    else:
        print("[WARNING] Disabled EMA model for test!")
        lightning_model = model

    # Define denoiser
    denoiser: SigmaDockDenoiser = lightning_model.model
    denoiser.eval()
    print(denoiser)
    print(f"Number of parameters: {sum(p.numel() for p in denoiser.parameters() if p.requires_grad)}")

    # Create sampler
    sampler_fn: Callable = partial(
        sampler,
        denoiser=denoiser,
        t_min=cfg.diffusion.t_min,
        rho=cfg.diffusion.rho,
        num_steps=cfg.diffusion.num_steps,
        noise_scale=cfg.diffusion.noise_scale,
        noise_decay=cfg.diffusion.noise_decay,
        solver=cfg.diffusion.solver,
        discretization=cfg.diffusion.discretization,
        use_true_scores=cfg.diffusion.use_true_scores,
        verbose=cfg.diffusion.verbose,
    )

    # Create Sampling Module
    sampling_module = SamplingModule(
        denoiser=denoiser,
        sampler=sampler_fn,
        seeds=seeds,
        cfg=cfg,
    )

    # Create inference trainer
    trainer = pl.Trainer(
        accelerator=cfg.hardware.accelerator,
        strategy=cfg.hardware.strategy,
        devices=cfg.hardware.devices,
        benchmark=True,
        inference_mode=True,
        deterministic="warn",
        logger=False,
        enable_checkpointing=False,
        max_epochs=1,
    )

    trainer.predict(
        sampling_module,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    print("start")
    sample()
