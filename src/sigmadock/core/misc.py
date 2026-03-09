import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def compute_decay_constant(lr_start: float, lr_end: float, max_epochs: int) -> float:
    """
    Returns the decay constant k such that:
        lr_end = lr_start * exp(-k * max_epochs)
    """
    # Avoid log(0) or negative ratio if lr_end >= lr_start
    # but typically for a decay: lr_end < lr_start
    return -math.log(lr_end / lr_start) / max_epochs


class StepDecayExponentialCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        # Step-based parameters
        min_lr_start: float,
        min_lr_end: float,
        max_lr_start: float,
        max_lr_end: float,
        max_steps: int,  # CHANGED from max_epochs
        n_cycles: int,
        warmup_frac: float = 0.2,
        last_step: int = -1,  # CHANGED from last_epoch
    ) -> None:
        self.min_lr_start = min_lr_start
        self.min_lr_end = min_lr_end
        self.max_lr_start = max_lr_start
        self.max_lr_end = max_lr_end

        self.max_steps = max_steps  # CHANGED
        self.n_cycles = n_cycles
        self.warmup_frac = warmup_frac

        # Precompute decay constants using max_steps
        self.k_min = compute_decay_constant(min_lr_start, min_lr_end, max_steps)  # CHANGED
        self.k_max = compute_decay_constant(max_lr_start, max_lr_end, max_steps)  # CHANGED

        # Cycle length is now in steps
        self.cycle_length = float(self.max_steps) / self.n_cycles  # CHANGED

        # Initialize the parent class using last_step
        super().__init__(optimizer, last_step)

    def get_lr(self) -> list[float]:
        # 1) global step
        t = max(self.last_epoch, 0)  # self.last_epoch is the internal step counter

        # 2) figure out which cycle we are in
        cycle_idx = int(t / self.cycle_length)
        cycle_idx = min(cycle_idx, self.n_cycles - 1)

        # 3) when did *this* cycle start (in steps)?
        t0 = cycle_idx * self.cycle_length

        # 4) freeze the endpoints as of step t0
        decayed_min_lr = self.min_lr_start * math.exp(-self.k_min * t0)
        decayed_max_lr = self.max_lr_start * math.exp(-self.k_max * t0)

        # 5) fraction through this cycle
        t_in_cycle = t - t0
        cycle_frac = min(t_in_cycle / self.cycle_length, 1.0)

        # 6) LR calculation logic (this part remains the same)
        if cycle_idx == 0:
            cos_f = 0.5 * (1.0 + math.cos(math.pi * cycle_frac))
            lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * cos_f
        else:
            if cycle_frac < self.warmup_frac:
                wp = cycle_frac / self.warmup_frac
                ramp = 1.0 - math.cos(wp * math.pi / 2)
                lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * ramp
            else:
                dp = (cycle_frac - self.warmup_frac) / (1.0 - self.warmup_frac)
                cos_f = 0.5 * (1.0 + math.cos(math.pi * dp))
                lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * cos_f

        return [lr for _ in self.base_lrs]


class DecayExponentialCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        # Instead of half-lives, we directly specify start/end LR values:
        min_lr_start: float,
        min_lr_end: float,
        max_lr_start: float,
        max_lr_end: float,
        # Scheduling parameters:
        max_epochs: int,
        n_cycles: int,
        warmup_frac: float = 0.2,
        last_epoch: int = -1,
    ) -> None:
        """
        Exponential + Cosine Warm Restarts:
          - min_lr(t) decays exponentially from min_lr_start -> min_lr_end over max_epochs
          - max_lr(t) decays exponentially from max_lr_start -> max_lr_end over max_epochs
          - We have n_cycles cycles total, each with warmup_frac fraction for warmup,
            then cosine decay for the remainder of the cycle.
          - The final epoch ends exactly at decayed_min_lr (the bottom of the cosine).

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            min_lr_start (float): Starting min_lr at epoch=0.
            min_lr_end (float): Target min_lr at epoch=max_epochs.
            max_lr_start (float): Starting max_lr at epoch=0.
            max_lr_end (float): Target max_lr at epoch=max_epochs.
            max_epochs (int): Total number of epochs for the schedule.
            n_cycles (int): Number of warm restart cycles.
            warmup_frac (float): Fraction of each cycle used for warmup. Defaults to 0.2.
            last_epoch (int): Index of the last epoch. Defaults to -1 (meaning "before training starts").
        """
        # Save base references
        self.min_lr_start = min_lr_start
        self.min_lr_end = min_lr_end
        self.max_lr_start = max_lr_start
        self.max_lr_end = max_lr_end

        self.max_epochs = max_epochs
        self.n_cycles = n_cycles
        self.warmup_frac = warmup_frac

        # Precompute decay constants
        self.k_min = compute_decay_constant(min_lr_start, min_lr_end, max_epochs)
        self.k_max = compute_decay_constant(max_lr_start, max_lr_end, max_epochs)

        # Float-based cycle length
        self.cycle_length = float(self.max_epochs) / self.n_cycles

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        # 1) global epoch
        t = max(self.last_epoch, 0)

        # 2) figure out which cycle we are in
        cycle_idx = int(t / self.cycle_length)
        # clamp to last cycle if you overshoot
        cycle_idx = min(cycle_idx, self.n_cycles - 1)

        # 3) when did *this* cycle start?
        t0 = cycle_idx * self.cycle_length

        # 4) freeze the endpoints as of t0
        decayed_min_lr = self.min_lr_start * math.exp(-self.k_min * t0)
        decayed_max_lr = self.max_lr_start * math.exp(-self.k_max * t0)

        # 5) fraction through this cycle
        t_in_cycle = t - t0
        cycle_frac = min(t_in_cycle / self.cycle_length, 1.0)

        # 6) pure cosine on cycle 0, warmup+cosine on cycles >= 1
        if cycle_idx == 0:
            # half cosine from max → min
            cos_f = 0.5 * (1.0 + math.cos(math.pi * cycle_frac))
            lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * cos_f

        else:
            if cycle_frac < self.warmup_frac:
                # warmu pramp: half cosine from min - max
                wp = cycle_frac / self.warmup_frac
                ramp = 1.0 - math.cos(wp * math.pi / 2)
                lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * ramp
            else:
                # cosinedecay: half cosine from max - min
                dp = (cycle_frac - self.warmup_frac) / (1.0 - self.warmup_frac)
                cos_f = 0.5 * (1.0 + math.cos(math.pi * dp))
                lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * cos_f

        # 7) one LR per param group
        return [lr for _ in self.base_lrs]

    def get_lr_deprecated(self) -> list[float]:
        # NOTE this implements inverse cosine decay
        # Current epoch index
        t = max(self.last_epoch, 0)

        # Exponentially decayed min and max at epoch t
        decayed_min_lr = self.min_lr_start * math.exp(-self.k_min * t)
        decayed_max_lr = self.max_lr_start * math.exp(-self.k_max * t)

        # Which cycle are we in?
        cycle_idx = int(t / self.cycle_length)
        # Float-based position in this cycle
        t_in_cycle = t - cycle_idx * self.cycle_length
        cycle_frac = t_in_cycle / self.cycle_length
        # Clamp final epoch fraction to 1.0
        cycle_frac = min(cycle_frac, 1.0)

        # Warmup length in fraction
        if cycle_idx == 0:
            # In the first cycle, let's do a direct cosine from decayed_max_lr -> decayed_min_lr
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_frac))
            lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * cosine_factor
        else:
            # For cycles >= 1, do warmup + decay
            if cycle_frac < self.warmup_frac:
                # Warmup from decayed_min_lr -> decayed_max_lr
                warmup_progress = cycle_frac / self.warmup_frac
                # Half-cosine ramp from 0->1
                ramp_factor = 1.0 - math.cos((warmup_progress * math.pi) / 2)
                lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * ramp_factor
            else:
                # Cosine decay from decayed_max_lr -> decayed_min_lr
                decay_progress = (cycle_frac - self.warmup_frac) / (1.0 - self.warmup_frac)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                lr = decayed_min_lr + (decayed_max_lr - decayed_min_lr) * cosine_factor

        return [lr for _ in self.base_lrs]

    def step(self, epoch: int | None = None) -> None:
        """
        Advances the scheduler by one step. If `epoch` is given, it sets the current epoch to `epoch`.
        """
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        super().step(self.last_epoch)


class ExponentialCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        min_lr: float,
        max_lr: float,
        max_epochs: int,
        n_cycles: int,
        half_life_frac: float,
        warmup_frac: float = 0.2,
        last_epoch: int = -1,
    ) -> None:
        """
        Implements cosine annealing with warm restarts under an exponential decay envelope.

        The scheduler:
        - Starts with a cosine decay from `max_lr` to `min_lr` in the first cycle.
        - For subsequent cycles, warms up from `min_lr` to a decayed peak learning rate.
        - Each cycle is shaped by a cosine annealing curve.
        - The peak learning rate follows an exponential decay with a user-defined half-life fraction.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            min_lr (float): Minimum learning rate.
            max_lr (float): Initial maximum learning rate.
            max_epochs (int): Total number of training epochs.
            n_cycles (int): Number of warm restart cycles across `max_epochs`.
            half_life_frac (float): Fraction of `max_epochs` at which the envelope decays by half.
            warmup_frac (float, optional): Fraction of each cycle used for warmup (default: 0.2).
            last_epoch (int, optional): Last epoch index (default: -1).
        """
        raise DeprecationWarning("This scheduler is deprecated. Use StepDecayExponentialCosineAnnealingWarmRestarts instead.")
        self.min_lr: float = min_lr
        self.max_lr: float = max_lr
        self.max_epochs: int = max_epochs
        self.n_cycles: int = n_cycles
        self.half_life_frac: float = half_life_frac
        self.cycle_length: int = max_epochs // n_cycles
        self.warmup_length: int = int(self.cycle_length * warmup_frac)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """
        Computes the learning rate for the current epoch.

        Returns:
            list[float]: Learning rate for each parameter group.
        """
        t = max(self.last_epoch, 0)
        decayed_min = self.min_lr_start * math.exp(-self.k_min * t)
        decayed_max = self.max_lr_start * math.exp(-self.k_max * t)

        cycle_idx = int(t / self.cycle_length)
        t_in_cycle = t - cycle_idx * self.cycle_length
        cycle_frac = min(t_in_cycle / self.cycle_length, 1.0)

        if cycle_frac < self.warmup_frac:
            # Warmup: ramp from min → max
            wp = cycle_frac / self.warmup_frac
            rf = 1.0 - math.cos(wp * math.pi / 2)
            lr = decayed_min + (decayed_max - decayed_min) * rf
        else:
            # Cosine decay: from max → min
            dp = (cycle_frac - self.warmup_frac) / (1.0 - self.warmup_frac)
            cf = 0.5 * (1.0 + math.cos(math.pi * dp))
            lr = decayed_min + (decayed_max - decayed_min) * cf

        return [lr for _ in self.base_lrs]

    def step(self, epoch: int | None = None) -> None:
        """
        Advances the scheduler by one step.

        Args:
            epoch (int, optional): Manually specify the current epoch (default: None).
        """
        self.last_epoch = epoch if epoch is not None else self.last_epoch + 1
        super().step(self.last_epoch)
