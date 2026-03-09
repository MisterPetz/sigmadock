import sys
import traceback
from copy import deepcopy
from itertools import islice
from typing import Any, Optional

import torch
import torch.distributed as dist
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn


class SetEpochCallback(Callback):
    """
    A PyTorch Lightning callback to manually call `set_epoch` on an
    IterableDataset at the start of each training epoch.

    This is a robust workaround for cases where Lightning's automatic
    mechanism fails.
    """

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Get the underlying dataset from the DataLoader trainer.train_dataloader is a dictionary in this context
        dataset = trainer.train_dataloader.dataset

        # Check if the dataset has a `set_epoch` method
        if hasattr(dataset, "set_epoch"):
            # Call it with the current epoch number
            dataset.set_epoch(trainer.current_epoch)
            print(f"[DEBUG] Manually set epoch to {trainer.current_epoch} on rank {trainer.global_rank}")
        else:
            # If the dataset does not have a `set_epoch` method, raise an error
            raise MisconfigurationException(
                "The dataset does not have a `set_epoch` method. "
                "Ensure you are using an IterableDataset with set_epoch implemented."
            )


class SamplerDebugCallback(Callback):
    """
    Print the first `num_indices` indices yielded by the train sampler at
    the start of each epoch — one line per rank. Useful to verify:
      - that DistributedSampler is present,
      - that different ranks get disjoint shards within an epoch,
      - and that indices change across epochs (i.e. sampler.set_epoch is effective).
    """

    def __init__(self, num_indices: int = 10) -> None:
        super().__init__()
        self.num_indices = num_indices

    def _get_sampler(self, dl: Any) -> Optional[object]:
        # Try the usual attributes where the sampler can live
        s = getattr(dl, "sampler", None)
        if s is None:
            bs = getattr(dl, "batch_sampler", None)
            if bs is not None:
                # some DataLoaders wrap sampler in batch_sampler
                s = getattr(bs, "sampler", None) or bs
        return s

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # trainer.train_dataloader might be a single DataLoader or a list
        dl = trainer.train_dataloader
        if isinstance(dl, (list, tuple)):
            dl = dl[0]

        sampler = self._get_sampler(dl)
        rank = getattr(trainer, "global_rank", 0)
        epoch = getattr(trainer, "current_epoch", 0)

        if sampler is None:
            print(f"[R{rank}] epoch={epoch} sampler=None")
            return

        # Try to iterate only the first few indices (avoid building the whole list)
        try:
            it = iter(sampler)
            first = list(islice(it, self.num_indices))
        except Exception as e:
            first = f"<error iterating sampler: {e}>"

        print(f"[R{rank}] epoch={epoch} sampler_type={type(sampler).__name__} first_indices={first}")


class EMAWithRampup(Callback):
    def __init__(
        self,
        ema_halflife_kpoints: float,
        ema_rampup_ratio: Optional[float] = None,
        batch_size: int = 64,
        update_every: int = 1,
        sync_every: int = 10,
        cold_steps: int = 0,
        use_ema_for_val: bool = False,
    ) -> None:
        super().__init__()
        self.ema_halflife_kpoints = ema_halflife_kpoints
        self.ema_rampup_ratio = ema_rampup_ratio
        self.batch_size = batch_size
        self.update_every = update_every
        self.sync_every = sync_every
        self.use_ema_for_val = use_ema_for_val
        self.cold_steps = cold_steps

        self.steps = 0
        self._backup: dict[str, torch.Tensor] = {}
        self.ema_model = None

    def state_dict(self) -> dict[str, Any]:
        return {"steps": self.steps}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.steps = state_dict["steps"]

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.ema_model is None:
            print(f"Creating a shadow EMA model on rank {trainer.global_rank} in the validation loop.")
            # create a frozen "shadow" copy
            self.ema_model = deepcopy(pl_module.model).to(pl_module.device).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.ema_model is None:
            print(f"Creating a shadow EMA model on rank {trainer.global_rank} in the training loop.")
            # create a frozen "shadow" copy
            self.ema_model = deepcopy(pl_module.model).to(pl_module.device).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.steps += 1
        if self.steps < self.cold_steps:
            return

        if (self.steps % self.update_every == 0) and (self.steps > 0):
            self._update_ema(pl_module)

            # sync every sync_every updates
            if (
                dist.is_available()
                and dist.is_initialized()
                and (self.steps // self.update_every) % self.sync_every == 0
            ):
                self._sync_ema()

    @torch.no_grad()
    def _update_ema(self, pl_module: LightningModule) -> None:
        # convert half-life
        N = self.ema_halflife_kpoints * 1000 / self.update_every
        if self.ema_rampup_ratio is not None:
            N = min(N, self.steps * self.batch_size * self.ema_rampup_ratio)
        beta = 0.5 ** (self.batch_size / max(N, 1e-8))

        for p_ema, p_cur in zip(self.ema_model.parameters(), pl_module.model.parameters()):
            p_ema.data.lerp_(p_cur.detach().data, 1.0 - beta)

        pl_module.log("train/ema_beta", beta, on_step=True, prog_bar=False)

    @torch.no_grad()
    def _sync_ema(self) -> None:
        # Fuse all EMA params into one tensor and average across ranks
        params = list(self.ema_model.parameters())
        # flatten
        flat = torch.cat([p.data.view(-1) for p in params])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(dist.get_world_size())
        # unflatten
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(flat[offset : offset + numel].view_as(p.data))
            offset += numel

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.use_ema_for_val:
            if dist.is_available() and dist.is_initialized():
                # Synchronize EMA parameters across DDP workers, if applicable.
                self._sync_ema()
            self._backup = {k: v.clone() for k, v in pl_module.model.state_dict().items()}
            pl_module.model.load_state_dict(self.ema_model.state_dict())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.use_ema_for_val and self._backup:
            pl_module.model.load_state_dict(self._backup)
            self._backup = {}

    @rank_zero_only
    def compute_ema_drift(self, pl_module: LightningModule) -> float:
        total_diff = 0.0
        total_norm = 0.0
        for p_cur, p_ema in zip(pl_module.model.parameters(), self.ema_model.parameters()):
            diff = (p_cur - p_ema).pow(2).sum()
            norm = p_cur.pow(2).sum()
            total_diff += diff
            total_norm += norm
        return (total_diff / (total_norm + 1e-8)).sqrt().item()

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        drift = self.compute_ema_drift(pl_module=pl_module)
        pl_module.log("train/ema_drift", drift, on_epoch=True)

    @rank_zero_only
    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, ckpt: dict) -> None:
        ckpt["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, ckpt: dict) -> None:
        # If we havent yet built the shadow model, do it now
        if self.ema_model is None:
            # this runs before on_load_checkpoint
            self.ema_model = deepcopy(pl_module.model).to(pl_module.device).eval()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)
        # Load the EMA state dict if it exists in the checkpoint
        if "ema_state_dict" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_state_dict"])


class EMAWithRampupEpochUpdate(Callback):
    def __init__(
        self,
        ema_model: nn.Module,
        model: nn.Module,
        ema_halflife_kpoints: float,
        ema_rampup_ratio: Optional[float] = None,
        batch_size: int = 1,
        update_every: int = 1,
        cold_steps: int = 0,
        use_ema_for_val: bool = False,
    ) -> None:
        """
        EMA callback that updates an EMA copy of model parameters with a half-life defined in kimg (thousands of images)
        The update starts only after `cold_steps` training steps.

        Args:
            ema_model (nn.Module): The EMA model (should have the same architecture as model).
                                   This is the "shadow" model whose parameters are updated.
            model (nn.Module): The original model whose parameters are used for EMA updates.
            ema_halflife_kpoints (float): Half-life in thousands of datapoints.
            ema_rampup_ratio (Optional[float]): Ratio for ramp-up in early training.
                                    Effective half-life becomes min(static_half_life, step * batch_size * rampup_ratio).
            batch_size (int): Batch size used in training.
            update_every (int): Update EMA only every this many training steps.
            use_ema_for_val (bool): If True, the models weights are temporarily replaced with EMA weights during
                validation.
            cold_steps (int): Number of initial training steps during which EMA is not updated.
        """
        super().__init__()
        self.ema_model = ema_model
        self.model = model
        self.ema_halflife_kpoints = ema_halflife_kpoints
        self.ema_rampup_ratio = ema_rampup_ratio
        self.batch_size = batch_size
        self.update_every = update_every
        self.use_ema_for_val = use_ema_for_val
        self.cold_steps = cold_steps
        self.steps = 0  # Global training step counter
        self._backup: dict[str, torch.Tensor] = {}

    def state_dict(self) -> dict[str, Any]:
        return {"steps": self.steps}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.steps = state_dict["steps"]

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Create & move EMA model to the correct device
        self.ema_model = deepcopy(pl_module.model).eval()
        self.ema_model.to(pl_module.device)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        self.steps += 1
        if self.steps < self.cold_steps:
            return  # Skip EMA updates during cold steps.
        if self.steps % self.update_every == 0:
            self.update_ema(trainer=trainer)

    @torch.no_grad()
    def update_ema(self, trainer: Trainer | None) -> None:
        # Convert half-life from kimg to number of images.
        ema_halflife_npoints: float = self.ema_halflife_kpoints * 1000
        # Apply ramp-up: during early training, use a shorter effective half-life.
        if self.ema_rampup_ratio is not None:
            ema_halflife_npoints = min(ema_halflife_npoints, self.steps * self.batch_size * self.ema_rampup_ratio)
        # Compute the decay factor (ema_beta):
        # This gives: decay = 0.5^(batch_size / effective_half_life_in_images)
        ema_beta: float = 0.5 ** (self.batch_size / max(ema_halflife_npoints, 1e-8))

        # Update EMA model parameters using linear interpolation.
        for p_ema, p_net in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.data.lerp_(p_net.detach().data, 1 - ema_beta)

        if trainer is not None:
            trainer.logger.experiment.log({"train/ema_beta": ema_beta, "step": trainer.global_step})

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # If using EMA for validation, backup the current model and load EMA weights.
        if self.use_ema_for_val:
            self._backup = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.model.load_state_dict(self.ema_model.state_dict())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Synchronize EMA parameters across DDP workers, if applicable.
        if dist.is_available() and dist.is_initialized():
            for param in self.ema_model.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= dist.get_world_size()
        # Restore the original model weights after validation.
        if self.use_ema_for_val and self._backup:
            self.model.load_state_dict(self._backup)
            self._backup = {}

    def compute_ema_drift(self) -> float:
        """Noramlized Drift computation for EMA. Typical values below 0.1 are stable.

        Returns:
            float: normalized drift [0, 1]. Should aim for values around 0-0.1.
        """
        total_diff = 0.0
        total_norm = 0.0
        for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
            if p.requires_grad:
                diff = (p - ema_p).pow(2).sum()
                norm = p.pow(2).sum()
                total_diff += diff
                total_norm += norm
        drift = (total_diff / (total_norm + 1e-8)).sqrt().item()
        return drift

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        drift = self.compute_ema_drift()
        trainer.logger.experiment.log({"train/ema_drift": drift, "step": trainer.global_step})

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]) -> None:
        checkpoint["ema_state_dict"] = self.ema_model.state_dict()

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]) -> None:
        if "ema_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_state_dict"])


class FullNaNCheckCallback(Callback):
    """
    1) Enables torch.autograd anomaly detection so you get the exact
       autograd.Function where the NaN/Inf was first created.
    2) Hooks *before* sanity checking (on_pretrain_routine_start), scans both
       inputs and outputs of every module, and raises with the modules
       full name and the offending tensors shape.
    """

    def __init__(
        self,
        module_whitelist: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
    ) -> None:
        super().__init__()
        self.whitelist = module_whitelist

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa
        # Enable anomaly detection for backward-side tracebacks
        torch.autograd.set_detect_anomaly(True)

        name_map = {m: name for name, m in pl_module.named_modules()}

        def _pre_hook(module, inputs):  # noqa
            mod_name = name_map.get(module, module.__class__.__name__)
            # sys.stderr.write(f"[DEBUG] pre-hook {mod_name}\n")
            # sys.stderr.flush()

            for idx, t in enumerate(inputs):
                if isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any()):
                    stack = "".join(traceback.format_stack())
                    sys.stderr.write(
                        f"[DEBUG] pre-hook {mod_name} INPUT#{idx} NaN/Inf detected in {mod_name} (shape={tuple(t.shape)})\n"  # noqa
                    )
                    sys.stderr.flush()
                    raise MisconfigurationException(
                        f"NaN/Inf in forward INPUT#{idx} of {mod_name} shape={tuple(t.shape)}\n{stack}"
                    )

        for m in pl_module.modules():
            m.register_forward_pre_hook(_pre_hook)

        def _forward_hook(module: nn.Module, inputs: tuple[Any], outputs: Any) -> None:
            # skip if not in whitelist
            if self.whitelist is not None and not isinstance(module, self.whitelist):
                return

            module_name = name_map.get(module, module.__class__.__name__)

            def scan_tensor(t: torch.Tensor, where: str) -> None:
                if torch.isnan(t).any() or torch.isinf(t).any():
                    stack = "".join(traceback.format_stack())
                    sys.stderr.write(
                        f"[DEBUG] NaN/Inf in {where} of '{module_name}' (shape={tuple(t.shape)})\nCall stack:\n{stack}"
                    )
                    sys.stderr.flush()
                    raise MisconfigurationException(
                        f"\n NaN/Inf in {where} of '{module_name}' (shape={tuple(t.shape)})\nCall stack:\n{stack}"
                    )

            # 1) scan inputs
            for idx, x in enumerate(inputs):
                if isinstance(x, torch.Tensor):
                    scan_tensor(x, f"forward INPUT#{idx}")

            # 2) scan outputs
            outs = []
            if isinstance(outputs, torch.Tensor):
                outs = [outputs]
            elif isinstance(outputs, (list, tuple)):
                outs = [o for o in outputs if isinstance(o, torch.Tensor)]
            elif isinstance(outputs, dict):
                outs = [o for o in outputs.values() if isinstance(o, torch.Tensor)]

            for idx, t in enumerate(outs):
                scan_tensor(t, f"forward OUTPUT#{idx}")

        def _backward_hook(module, grad_inputs, grad_outputs):  # noqa
            # only scan if in whitelist (or None for all modules)
            if self.whitelist is not None and not isinstance(module, self.whitelist):
                return
            name = name_map[module]
            for idx, g in enumerate(grad_outputs):
                if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                    raise RuntimeError(f"NaN/Inf in backward grad_output#{idx} of `{name}` (shape={tuple(g.shape)})")
            for idx, g in enumerate(grad_inputs):
                if g is not None and (torch.isnan(g).any() or torch.isinf(g).any()):
                    raise RuntimeError(f"NaN/Inf in backward grad_input#{idx} of `{name}` (shape={tuple(g.shape)})")

