import sys
from typing import Any, Literal

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch import optim
from torch.nn.utils import clip_grad_norm_  # noqa
from torch.optim import lr_scheduler
from torch_geometric.data import Batch

from sigmadock.core.loaders import batch_is_empty

# from torch_geometric.utils import scatter
from sigmadock.core.misc import (
    DecayExponentialCosineAnnealingWarmRestarts,
    StepDecayExponentialCosineAnnealingWarmRestarts,
)
from sigmadock.diff.denoiser import SigmaDockDenoiser


class SigmaLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        denoiser: SigmaDockDenoiser,
        # Loss scalings
        fragment_scaling: float = 0.5,
        trans_score_weight: float = 1.0,
        rot_score_weight: float = 1 / 2,
        trans_data_weight: float = 0.0,
        rot_data_weight: float = 0.0,
        # Optimizer & Scheduler
        weight_decay: float = 1e-4,
        optimizer_eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.999),  # AdamW betas
        max_steps: int = 1e6,
        num_warmup_steps: int | float = 5e4,
        num_lr_cycles: int = 8,
        init_lr_start: float = 1e-6,
        min_lr_start: float = 4e-5,
        max_lr_start: float = 32e-5,
        min_lr_end: float = 1e-5,
        max_lr_end: float = 8e-5,
        cycle_warmup_frac: float = 1 / 4,
        grad_clip: float = 1.0,
        # Logging
        log_intermediates: bool = False,
        # Config (for loading from checkpoint)
        denoiser_config: dict[str, Any] | None = None,
        equiformer_config: dict[str, Any] | None = None,
        # Hardware
        compile: bool = False,
        # -----------------
        # Other ignored args (logged still)
        **kwargs: dict[str, Any],
    ) -> None:
        """SigmaDock Trainer Inputs

        Args:
            denoiser (nn.Module): The underlying Preconditioned Denoiser that handles forward passes.
            fragment_scaling (float): Scaling factor for fragment-based losses.
                Value of 0 = Average per fragment in the batch (biases large fragments).
                Value of 1 = Average per molecule equally across all fragments (biases small fragments).
                Value of -1 = No scaling (biases large fragments).
                Defaults to 0.5.
            trans_score_weight (float): Weight for translation score loss.
            rot_score_weight (float): Weight for rotation score loss.
            trans_data_weight (float): Weight for translation data loss.
            rot_data_weight (float): Weight for rotation data loss.
            weight_decay (float): Weight decay for the optimizer.
            optimizer_eps (float): Epsilon for the optimizer.
            max_epochs (int): Total number of epochs for scheduling.
            num_warmup_epochs (int): (Not used explicitly in this example, but kept for logging.)
            num_lr_cycles (int): Number of warm-restart cycles for the scheduler.
            init_lr_start (float): Starting initial LR at epoch 0.
            min_lr_start (float): Starting minimum LR at epoch 0.
            max_lr_start (float): Starting maximum LR at epoch 0.
            min_lr_end (float): Ending minimum LR at epoch = max_epochs.
            max_lr_end (float): Ending maximum LR at epoch = max_epochs.
            cycle_warmup_frac (float): Fraction of each cycle to use for warmup.
            kwargs: dict[str, Any] (Any): Additional arguments (logged but not directly used).
        """

        super().__init__(**kwargs)

        if compile:
            try:
                self.model = torch.compile(denoiser)
            except Exception as e:
                print(f"[WARN] Failed to compile model: {e}. Continuing without compilation.")
                self.model = denoiser
        else:
            self.model = denoiser

        # Scheduler args & kwargs
        self.init_lr_start = init_lr_start
        self.min_lr_start = min_lr_start
        self.min_lr_end = min_lr_end
        self.max_lr_start = max_lr_start
        self.max_lr_end = max_lr_end
        self.max_steps = max_steps
        self.num_warmup_steps = (
            num_warmup_steps if isinstance(num_warmup_steps, int) else int(num_warmup_steps * max_steps)
        )
        self.num_lr_cycles = num_lr_cycles
        self.cycle_warmup_frac = cycle_warmup_frac

        # Optimizer
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.optimizer_eps = optimizer_eps
        self.betas = betas

        # Loss scalings
        assert 0 <= fragment_scaling <= 1, "Fragment scaling must be between 0 and 1."
        self.fragment_scaling = fragment_scaling
        # Score scalings
        self.trans_score_weight = trans_score_weight
        self.rot_score_weight = rot_score_weight
        # Data scalings
        self.trans_data_weight = trans_data_weight
        self.rot_data_weight = rot_data_weight
        # Hparams
        # TODO ensure model is persistent. Currently old checkpoint will not load under new __class__
        self.save_hyperparameters(ignore=["denoiser"])

        # Logging & Monitoring
        self.log_intermediates = log_intermediates
        self.train_step_counter = 0
        self.val_step_counter = 0
        self.test_step_counter = 0
        self.epoch_counter = 0
        self.epoch_frames = []

    def configure_optimizers(self) -> None:
        opt = optim.AdamW(
            self.model.parameters(),
            lr=self.max_lr_start,
            weight_decay=self.weight_decay,
            eps=self.optimizer_eps,
            betas=self.betas,
        )
        # Use StepWise for more continuous control over the learning rate.
        scheduler = lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                lr_scheduler.LinearLR(
                    opt,
                    start_factor=(self.init_lr_start / self.max_lr_start),
                    total_iters=self.num_warmup_steps,
                ),
                StepDecayExponentialCosineAnnealingWarmRestarts(
                    opt,
                    min_lr_start=self.min_lr_start,
                    min_lr_end=self.min_lr_end,
                    max_lr_start=self.max_lr_start,
                    max_lr_end=self.max_lr_end,
                    max_steps=self.max_steps - self.num_warmup_steps,
                    n_cycles=self.num_lr_cycles,
                    warmup_frac=self.cycle_warmup_frac,
                ),
            ],
            milestones=[self.num_warmup_steps],
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        # Deprecated (Epoch-based scheduler)
        if False:
            scheduler = lr_scheduler.SequentialLR(
                opt,
                schedulers=[
                    lr_scheduler.LinearLR(
                        opt,
                        start_factor=(self.init_lr_start / self.max_lr_start),
                        total_iters=self.num_warmup_steps,
                    ),
                    DecayExponentialCosineAnnealingWarmRestarts(
                        opt,
                        min_lr_start=self.min_lr_start,
                        min_lr_end=self.min_lr_end,
                        max_lr_start=self.max_lr_start,
                        max_lr_end=self.max_lr_end,
                        max_epochs=self.max_epochs - self.num_warmup_epochs,
                        n_cycles=self.num_lr_cycles,
                        warmup_frac=self.cycle_warmup_frac,
                    ),
                ],
                milestones=[self.num_warmup_epochs],
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler_config}

    def forward(
        self, batch: Batch, *args: Any, **kwargs: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass for base nn.Module denoiser model.
        Args:
            *args: Positional arguments for the model.
            **kwargs: Keyword arguments for the model.
        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: The denoised outputs and losses.
        """
        denoised_outputs = self.model(batch, *args, **kwargs)
        losses = self.model.compute_losses(denoised_outputs)  # [B x F]
        for key, loss in losses.items():
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                raise ValueError(f"NaN/Inf in loss {key} with value {loss}")

        # NOTE if self.fragment_scaling == 1 we are not scaling the loss just computing average per mol.
        losses = self.model.scaled_fragmented_loss(
            losses=losses,
            num_fragments=denoised_outputs["num_fragments"],
            fragment_scaling=self.fragment_scaling,
        )  # [B]
        return denoised_outputs, losses

    def compute_grad_norm(self) -> float:
        """Compute the global L2 norm of all gradients."""
        norms = [p.grad.detach().norm(2) for p in self.parameters() if p.grad is not None]
        if not norms:
            return 0.0
        return torch.norm(torch.stack(norms), 2).item()

    def on_after_backward(self) -> None:
        # This hook is for VERIFICATION ONLY.
        # It calculates and logs the raw gradient norm before Lightning clips it.
        grad_norm_unclipped = self.compute_grad_norm()
        self.log("train/grad_norm_unclipped", grad_norm_unclipped, on_step=True, on_epoch=False)

        # --- DDP Unused Parameter Debugging ---
        # This will only run on the main process (rank 0) to avoid spamming the log.
        if self.trainer.global_rank == 0:
            # Find all parameters that did NOT receive a gradient
            unused_params = [name for name, p in self.model.named_parameters() if p.grad is None]

            if unused_params:
                print(f"--- Unused parameters on step {self.global_step}: ---")
                for name in unused_params:
                    print(name)
                print("--------------------------------------------------")

    def _shared_step(
        self, batch: Batch, batch_idx: int, stage: Literal["train", "val", "test"], **kwargs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        score_terms, losses = self(batch, **kwargs)
        for key, loss in losses.items():
            isnan = torch.isnan(loss)
            isinf = torch.isinf(loss)
            if isnan.any() or isinf.any():
                print(f"[WARN] NaN/Infs in loss {key} with values {loss}, removing from loss dict.")
                loss[isnan] = 0
                loss[isinf] = 0

        # Reduce losses into total loss according to weightings.
        total_loss = (
            self.trans_score_weight * losses["T_score"]
            + self.rot_score_weight * losses["R_score"]
            + self.trans_data_weight * losses["T0"]
            + self.rot_data_weight * losses["R0"]
        )

        # --- unpacking ---
        force_per_atom = score_terms["pseudoforces"]  # [B*F*A, 3]
        force_per_fragment = score_terms["force_per_fragment"]  # [B x F, 3]
        torque_per_fragment = score_terms["torque_per_fragment"]  # [B x F, 3]

        p_trans_score = score_terms["pred_T_score"]  # [B, F, 3]
        p_rot_score = score_terms["pred_R_score"]  # [B, F, 3, 3]
        t_trans_score = score_terms["true_T_score"]  # [B, F, 3]
        t_rot_score = score_terms["true_R_score"]  # [B, F, 3, 3]

        # Only sync scalars if test or val.
        log_dict: dict[str, float | torch.Tensor] = {
            f"loss_{stage}/total": total_loss.mean(),
            f"loss_{stage}/T_score": losses["T_score"].mean(),
            f"loss_{stage}/R_score": losses["R_score"].mean(),
            f"loss_{stage}/T0": losses["T0"].mean(),
            f"loss_{stage}/R0": losses["R0"].mean(),
        }

        # compute batch size (number of graphs)
        if hasattr(batch, "num_graphs"):
            batch_n = int(batch.num_graphs)
        elif hasattr(batch, "batch") and isinstance(batch.batch, torch.Tensor):
            batch_n = int(batch.batch.numel())
        else:
            batch_n = 0

        self.log_dict(
            log_dict,
            prog_bar=False,
            rank_zero_only=True,  # Log only on rank 0.
            on_step=(stage == "train"),  # Log on step for train.
            on_epoch=(stage != "train"),  # Log on epoch for train, val, test.
            sync_dist=(stage != "train"),  # Sync only on val/test.
            batch_size=batch_n,
        )

        # Check if collection should happen based on global_step and log_every_n_steps for train OR all val/test.
        if (
            self.logger
            and ((self.global_step % max(1, self.trainer.log_every_n_steps) == 0) or stage != "train")
            and self.log_intermediates
        ):
            # Convert tensors to scalars before collection
            intermediates = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in score_terms.items()}
            getattr(self, f"{stage}_step_outputs").append(intermediates)

        return log_dict

    def training_step(self, batch: Batch, batch_idx: int, **kwargs: dict[str, Any]) -> Any:
        stage = "train"
        self.train_step_counter += 1

        if batch_is_empty(batch):
            print(f"[WARN] Empty training batch at idx={batch_idx} — returning zeroed fake loss to keep DDP happy.")
            # optional logging so we can detect how often this happens
            self.log("train/num_empty_batches", 1, on_step=True, on_epoch=True, sync_dist=True)
            fake_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return fake_loss

        loss_dict: dict[str, torch.Tensor] = self._shared_step(
            batch=batch,
            batch_idx=batch_idx,
            stage="train",
            **kwargs,
        )
        total_loss = loss_dict[f"loss_{stage}/total"]

        # Return None if total_loss is NaN or Inf (safety check to not kill DDP)
        if not torch.isfinite(total_loss).all():
            # log warning and return fake loss to keep DDP alive
            sys.stderr.write(f"[WARN] Non-finite training loss detected: {total_loss}. Returning fake loss.\n")
            any_param = next(self.model.parameters())
            return any_param.sum() * 0.0

        # Log the loss
        return total_loss

    def validation_step(self, batch: Batch, batch_idx: int, **kwargs: dict[str, Any]) -> Any:
        stage = "val"
        self.val_step_counter += 1

        # For validation we can skip by returning None (Lightning allows this).
        if batch_is_empty(batch):
            # optional debug log
            print(f"[WARN] Empty validation batch at idx={batch_idx} — skipping.")
            return None

        loss_dict: dict[str, torch.Tensor] = self._shared_step(
            batch=batch,
            batch_idx=batch_idx,
            stage="val",
            **kwargs,
        )
        total_loss = loss_dict[f"loss_{stage}/total"]

        # safety: if non-finite, skip returning the bad value
        if not torch.isfinite(total_loss).all():
            print(f"[WARN] Non-finite validation loss at idx={batch_idx}: {total_loss}, skipping.")
            return None
        return total_loss

    def on_fit_start(self) -> None:
        device = next(self.model.parameters()).device
        self.model.diffuser._so3_diffuser.set_device(device)

    def on_train_epoch_start(self) -> None:
        self.train_step_counter = 0
        self.train_step_outputs = []

    def on_validation_epoch_start(self) -> None:
        self.val_step_counter = 0
        self.val_step_outputs = []

    def on_train_epoch_end(self) -> None:
        self._on_shared_epoch_end("train")
        self.epoch_counter += 1

    def on_validation_epoch_end(self) -> None:
        self._on_shared_epoch_end("val")

    def _on_shared_epoch_end(self, stage: Literal["train", "val", "test"]) -> None:
        # NOTE skipping sampling during training for speed.
        # TODO implement a sampling callback to run at the end of the epoch instead.
        return