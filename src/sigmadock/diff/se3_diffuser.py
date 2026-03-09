from pathlib import Path

import torch

from sigmadock.diff.r3_diffuser import R3Diffuser
from sigmadock.diff.so3_diffuser import SO3Diffuser

"""Adapted from https://github.com/jasonkyuyim/se3_diffusion."""


class SE3Diffuser:
    """Diffusion model for SE(3) using R3 and SO(3) diffusion steps.
    This class implements the diffusion process for SE(3) using R3 and SO(3) diffusion models.

    The diffusion process is defined by the following parameters:
        min_beta: minimum value of beta.
        max_beta: maximum value of beta.
        schedule: schedule for the diffusion process.
        min_sigma: minimum value of sigma.
        max_sigma: maximum value of sigma.
        num_sigma: number of sigma values.
        num_omega: number of omega values.
        cache_path: path to the cache directory.
        use_cached_score: whether to use cached score.
        L: Axis-angle expansion length.
    """

    def __init__(
        self,
        # r3 parameters
        min_beta: float,
        max_beta: float,
        # so3 parameters
        schedule: str,
        min_sigma: float,
        max_sigma: float,
        num_sigma: int,
        num_omega: int,
        cache_path: str | Path,
        use_cached_score: bool = True,
        L: int = 1000,
    ) -> None:
        # VE, Continuous
        self._so3_diffuser = SO3Diffuser(
            schedule=schedule,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            num_omega=num_omega,
            cache_path=Path(cache_path),
            use_cached_score=use_cached_score,
            L=L,
        )
        # VP, Fixed Steps
        self._r3_diffuser = R3Diffuser(min_beta=min_beta, max_beta=max_beta)

    def forward_marginal(self, trans_0: torch.Tensor, R_0: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns the noised translations and rotations at time t."""
        # trans_0: [n, 3], R_0: [n, 3, 3], t: [n]
        # Keep higher precision for the forward marginal calculations.
        trans_t, trans_score = self._r3_diffuser.forward_marginal(trans_0, t)  # [n, 3]
        trans_score_scaling = self._r3_diffuser.score_scaling(t)  # [n]

        R_t, rot_score = self._so3_diffuser.forward_marginal(R_0, t)  # [n, 3, 3]
        rot_score_scaling = self._so3_diffuser.score_scaling(t)

        return {
            "T_t": trans_t,
            "R_t": R_t,
            "T_score": trans_score,
            "R_score": rot_score,
            "T_score_scaling": trans_score_scaling,
            "R_score_scaling": rot_score_scaling,
        }

    def calc_trans_0(self, trans_score: torch.Tensor, trans_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Calculates E(x_0|x_t) from the score."""
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(self, trans_t: torch.Tensor, trans_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self._r3_diffuser.score(trans_t, trans_0, t)

    def calc_rot_score(self, R_t: torch.Tensor, R_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns conditional score as object in tangent space at R_t"""
        return self._so3_diffuser.score(R_t, R_0, t)

    def score(
        self, trans_t: torch.Tensor, trans_0: torch.Tensor, R_t: torch.Tensor, R_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans_score = self._r3_diffuser.score(trans_t, trans_0, t)
        rot_score = self._so3_diffuser.score(R_t, R_0, t)

        return trans_score, rot_score  # convention to have trans, R format

    def score_scaling(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        return {
            "T": trans_score_scaling,
            "R": rot_score_scaling,
        }

    def score_weight(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._r3_diffuser.score_weight(t), self._so3_diffuser.score_weight(t)

    def reverse(
        self,
        trans_t: torch.Tensor,
        R_t: torch.Tensor,
        trans_score: torch.Tensor,
        rot_score: torch.Tensor,
        t: float,
        dt: float,
        noise_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse sampling function from (t) to (t-1).

        Args:
            trans_t: [n, 3] translation at time t.
            R_t: [n, 3, 3] rotation at time t.
            rot_score: [n, 3, 3] rotation score.
            trans_score: [n, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            noise_scale: scaling factor for noise.

        Returns:
            trans_t_1, R_t_1: [n, 3] translation at time t-1,
            R_t_1: [n, 3, 3] rotation at time t-1.
        """

        trans_t_1 = self._r3_diffuser.reverse(x_t=trans_t, score_t=trans_score, t=t, dt=dt, noise_scale=noise_scale)
        R_t_1 = self._so3_diffuser.reverse(
            R_t=R_t,
            score_t=rot_score,
            t=t,
            dt=dt,
            noise_scale=noise_scale,
        )

        return trans_t_1, R_t_1

    def sample_ref(self, n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples rigids from reference distribution.

        Args:
            n: Number of samples.
            device: Device to put tensors on.
        """

        trans_ref = self._r3_diffuser.sample_ref(n=n, device=device)
        R_ref = self._so3_diffuser.sample_ref(n=n, device=device)

        return trans_ref, R_ref

    def sigma(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract sigma(t) corresponding to chosen sigma schedule."""

        sigma_R = self._so3_diffuser.sigma(t)
        sigma_T = self._r3_diffuser.sigma(t)

        return {
            "R": sigma_R,
            "T": sigma_T,
        }

    def score_weighting(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculates weight used for scores during training."""

        weight_R = self._so3_diffuser.score_weight(t)
        weight_T = self._r3_diffuser.score_weight(t)

        return {
            "R": weight_R,
            "T": weight_T,
        }


if __name__ == "__main__":
    se3_diffuser = SE3Diffuser(
        min_beta=0.1,
        max_beta=20,
        schedule="logarithmic",
        min_sigma=0.1,
        max_sigma=1.5,
        num_sigma=1000,
        num_omega=2000,
        cache_path=Path("/homes/lezhang/toy-dock/runs"),
        L=1000,
    )

    n = 5

    trans_0 = torch.randn(n, 3)
    R_0 = torch.eye(3).expand(n, 3, 3)

    print("\ntest forward_marginal")
    t = torch.rand(n)
    dict_t = se3_diffuser.forward_marginal(trans_0, R_0, t)
    print(dict_t.keys())
    trans_t = dict_t["trans_t"]
    R_t = dict_t["R_t"]

    print("\ntest score")
    print(se3_diffuser.score(trans_t, trans_0, R_t, R_0, t)[0].shape)

    print("\ntest reverse")
    trans_score = dict_t["trans_score"]
    rot_score = dict_t["rot_score"]
    t = t[0].item()
    dt = 0.1
    print(se3_diffuser.reverse(trans_t, R_t, trans_score, rot_score, t, dt)[1].shape)
