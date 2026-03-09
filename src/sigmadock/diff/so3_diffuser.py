from pathlib import Path

import numpy as np
import torch

# from torch.cuda.amp import custom_bwd, custom_fwd
from sigmadock.diff import so3_utils

"""Adapted from https://github.com/jasonkyuyim/se3_diffusion."""


def igso3_expansion(omega: torch.Tensor, eps: torch.Tensor, L: int = 1000) -> torch.Tensor:
    """Computes the truncated IGSO3 density for angles in [0, pi]."""
    # omega: [num_omega], eps: []
    # omega: [n], eps: [n]
    ls = torch.arange(L)
    ls = ls.to(omega.device)

    if eps.ndim == 0:
        # Used during cache computation.
        ls = ls[None]  # [1, L]
        omega = omega[:, None]  # [num_omega, 1]
    elif eps.ndim == 1:
        # Used during predicted score calculation.
        ls = ls[None]  # [1, L]
        omega = omega[:, None]  # [n, 1]
        eps = eps[:, None]  # [n, 1]
    else:
        raise ValueError("eps must be 0D or 1D.")

    p = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * eps**2 / 2) * torch.sin(omega * (ls + 1 / 2)) / torch.sin(omega / 2)
    return p.sum(dim=-1)  # [num_omega], [n]


def igso3_density(omega: torch.Tensor, eps: torch.Tensor, L: int = 1000, marginal: bool = True) -> torch.Tensor:
    """IGSO(3) density for angles in [0, pi]."""
    # expansion: [num_omega], eps: []
    expansion = igso3_expansion(omega, eps, L=L)
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1 - torch.cos(omega)) / np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / torch.pi**2


# @custom_fwd(cast_inputs=False)
# @custom_bwd
def d_f_igso3_d_omega(omega: torch.Tensor, eps: torch.Tensor, L: int = 1000) -> torch.Tensor:
    """
    Computes the derivative of f_igso3 with respect to omega.
    """
    # omega: [num_omega], eps: []
    # omega: [n], eps: [n]
    ls = torch.arange(L)
    ls = ls.to(omega.device)

    if eps.ndim == 0:
        # used for caching
        ls = ls[None]  # [1, L]
        omega = omega[..., None]  # [num_omega, 1]
    elif eps.ndim == 1:
        # used for score calculation
        ls = ls[None]  # [1, L]
        omega = omega[:, None]  # [n, 1]
        eps = eps[:, None]  # [n, 1]
    else:
        raise ValueError("eps must be 0D or 1D.")

    hi = torch.sin(omega * (ls + 1 / 2))
    dhi = (ls + 1 / 2) * torch.cos(omega * (ls + 1 / 2))
    lo = torch.sin(omega / 2)
    dlo = 1 / 2 * torch.cos(omega / 2)
    dSigma = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * eps**2 / 2) * (lo * dhi - hi * dlo) / lo**2
    dSigma = dSigma.sum(dim=-1)  # [num_omega], [n]
    return dSigma


# @custom_fwd(cast_inputs=False)
# @custom_bwd
def d_log_f_d_omega(omega: torch.Tensor, eps: torch.Tensor, L: int = 1000) -> torch.Tensor:
    # omega: [num_omega], eps: []
    # omega: [n], eps: [n]
    f = igso3_expansion(omega, eps, L=L)  # 1D
    d_f_dx = d_f_igso3_d_omega(omega, eps, L=L)
    # The approxmation below to the derivative at t<-1.3 is very close to the true value
    eps_2 = eps**2

    # Safe branching
    mask = eps_2 < 0.3

    # Only compute branch1 where mask is True
    safe_branch1 = torch.zeros_like(omega)
    if mask.any():
        safe_branch1[mask] = -omega[mask] / eps_2[mask]

    # Only compute branch2 where mask is False
    safe_branch2 = torch.zeros_like(omega)
    if (~mask).any():
        safe_branch2[~mask] = d_f_dx[~mask] / f[~mask]

    # Combine the two branches
    return safe_branch1 + safe_branch2
    # return torch.where(eps_2 < 0.3, -omega / eps_2, d_f_dx / f)


class SO3Diffuser:
    """VE-SDE diffuser class for rotations."""

    def __init__(
        self,
        schedule: str = "logarithmic",
        min_sigma: float = 0.01**0.5,
        max_sigma: float = 2.25**0.5,
        num_sigma: int = 1000,
        num_omega: int = 2000,
        cache_path: Path | str | None = None,
        use_cached_score: bool = True,
        L: int = 1000,
    ) -> None:
        self.schedule = schedule

        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma

        self.num_omega = num_omega

        self.cache_path = cache_path
        self.use_cached_score = use_cached_score

        self.L = L

        self.vbucket = torch.vmap(self.bucket, in_dims=(0, 0))

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.discrete_omega = torch.linspace(0, np.pi, num_omega + 1)[1:]
        self.discrete_sigma = self.sigma(torch.linspace(0.0, 1.0, self.num_sigma))

        cache_dir = self.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        pdf_cache = cache_dir / "pdf_vals.pt"
        cdf_cache = cache_dir / "cdf_vals.pt"
        score_norms_cache = cache_dir / "score_norms.pt"

        if pdf_cache.exists() and cdf_cache.exists() and score_norms_cache.exists():
            print(f"Using cached IGSO3 in {cache_dir}")
            self._pdf = torch.load(pdf_cache)
            self._cdf = torch.load(cdf_cache)
            self._score_norms = torch.load(score_norms_cache)
        else:
            print(f"Computing IGSO3. Saving in {cache_dir}")
            # [num_sigma, num_omega]
            self._pdf = torch.stack(
                [igso3_density(self.discrete_omega, x, L=self.L, marginal=True) for x in self.discrete_sigma]
            )
            raw_cdf = torch.stack([pdf.cumsum(0) / self.num_omega * np.pi for pdf in self._pdf])
            self._cdf = raw_cdf / raw_cdf[:, -1:].clamp(min=1e-12)  # safe divide
            self._score_norms = torch.stack(
                [d_log_f_d_omega(self.discrete_omega, x, L=self.L) for x in self.discrete_sigma]
            )
            # Cache the precomputed values
            torch.save(self._pdf, pdf_cache)
            torch.save(self._cdf, cdf_cache)
            torch.save(self._score_norms, score_norms_cache)

        # NOTE: comes from variance of the conditional SO(3) score
        # [num_sigma]
        self._score_scaling = torch.sqrt(
            torch.abs(torch.sum(self._score_norms**2 * self._pdf, axis=-1) / torch.sum(self._pdf, axis=-1))
        ) / np.sqrt(3)

    @property
    def cache_dir(self) -> Path:
        min_sigma = str(self.min_sigma).replace(".", "_")
        max_sigma = str(self.max_sigma).replace(".", "_")
        cache_dir_name = (
            f"eps_{self.num_sigma}_omega_{self.num_omega}_min_sigma_{min_sigma}_"
            f"max_sigma_{max_sigma}_schedule_{self.schedule}"
        )
        return self.cache_path / cache_dir_name

    def set_device(self, device: str) -> None:
        """Set device for cached tensors."""
        self._pdf = self._pdf.to(device)
        self._cdf = self._cdf.to(device)
        self._score_norms = self._score_norms.to(device)
        self._score_scaling = self._score_scaling.to(device)
        self.discrete_omega = self.discrete_omega.to(device)
        self.discrete_sigma = self.discrete_sigma.to(device)

    def sigma_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        # sigma: [n]
        return torch.bucketize(sigma, self.discrete_sigma, right=True) - 1  # [n]

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Extract sigma(t) corresponding to chosen sigma schedule."""
        if self.schedule == "logarithmic":
            sigma = torch.log(t * np.exp(self.max_sigma) + (1 - t) * np.exp(self.min_sigma))
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")
        return sigma  # [n]

    def diffusion_coef(self, t: torch.Tensor) -> torch.Tensor:
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == "logarithmic":
            g_t = torch.sqrt(
                2 * (np.exp(self.max_sigma) - np.exp(self.min_sigma)) * self.sigma(t) / torch.exp(self.sigma(t))
            )
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")
        return g_t  # [n]

    def t_to_idx(self, t: torch.Tensor) -> torch.Tensor:
        """Helper function to go from time t to corresponding sigma_idx."""
        return self.sigma_idx(self.sigma(t))  # [n]

    def bucket(self, x: torch.Tensor, cdf_t: torch.Tensor) -> torch.Tensor:
        # x: [], cdf_t: [num_omega]
        return torch.bucketize(x, cdf_t, right=True) - 1

    def sample_igso3_angle(self, t: torch.Tensor) -> torch.Tensor:
        # t: [n]
        n = len(t)
        cdf_t = self._cdf[self.t_to_idx(t)]  # [n, num_omega]

        x = torch.rand([n, 1], device=t.device)
        num_omega = cdf_t.shape[1]
        x_idx = self.vbucket(x, cdf_t)  # [n, 1]
        x_idx = torch.clamp(x_idx, min=0, max=num_omega - 2)  # Ensure x_idx + 1 < num_omega
        x0 = torch.gather(cdf_t, 1, x_idx)  # [n, 1]
        x1 = torch.gather(cdf_t, 1, x_idx + 1)

        repeated_discrete_omega = torch.repeat_interleave(self.discrete_omega[None], n, dim=0)
        y0 = torch.gather(repeated_discrete_omega, 1, x_idx)  # [n, 1]
        y1 = torch.gather(repeated_discrete_omega, 1, x_idx + 1)
        # Interpolate to get the angle
        denom = x1 - x0
        # avoid div-zero
        eps = 1e-8
        denom = torch.where(denom.abs() < eps, torch.ones_like(denom) * eps, denom)
        angles = y0 + (x - x0) * (y1 - y0) / denom
        return angles.squeeze(-1)  # [n]

    def sample(self, t: torch.Tensor) -> torch.Tensor:
        """Generates rotation vector(s) from IGSO(3) at sigma_t."""
        n = len(t)
        # Sample random axes and guard zero norm
        axes = torch.randn(n, 3, device=t.device)
        norms = torch.linalg.norm(axes, dim=-1, keepdims=True)
        # Replace near-zero norms with 1 to avoid division by zero
        safe_norms = torch.where(norms < 1e-6, torch.ones_like(norms), norms)
        axes = axes / safe_norms
        # Optionally, replace axes with [1,0,0] where the norm was near-zero
        axes = torch.where(norms < 1e-6, torch.tensor([1.0, 0.0, 0.0], device=axes.device), axes)

        # Sample angles from IGSO(3) & ensure they are finite
        angles = self.sample_igso3_angle(t)  # [n]
        assert torch.isfinite(angles).all(), (
            f"Non finite angles in sample_igso3_angle: {angles[~torch.isfinite(angles)]}"
        )

        rot_vecs = axes * angles[..., None]  # [n, 3]
        assert rot_vecs.shape[-1] == 3, f"Expected [...,3] shape, got {rot_vecs.shape}"
        assert torch.isfinite(rot_vecs).all(), f"NaN/Inf in rot_vecs: {rot_vecs[~torch.isfinite(rot_vecs)]}"

        R = so3_utils.exp(so3_utils.hat(rot_vecs))  # [n, 3, 3]
        assert torch.isfinite(R).all(), "Non finite values in sampled rotation matrices"
        return R

    def sample_ref(self, n: int, device: str) -> torch.Tensor:
        # NOTE: replace with sample uniform?
        return self.sample(torch.ones([n], device=device))  # [n, 3]

    # @custom_fwd(cast_inputs=False)
    # @custom_bwd
    def score(self, R_t: torch.Tensor, R_0: torch.Tensor, t: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Compute the conditional score of R_t|R_0 at time t."""
        # R_t: [n, 3, 3]
        # R_0: [n, 3, 3]
        # t: [n]
        # from the commutivity of IGSO3
        R_0t = R_0 if sample else torch.einsum("...ji,...jk->...ik", R_0, R_t)  # compute R_0^T R
        log_R_0t = so3_utils.log(R_0t)
        omega = so3_utils.Omega(R_0t)  # [n]

        sigma = self.sigma(t).to(omega.device)  # [n]
        d_log_f = d_log_f_d_omega(omega, sigma, L=self.L)  # [n]

        score_R = torch.einsum("...ij,...jk->...ik", R_t, log_R_0t)
        score_R *= d_log_f[..., None, None] / omega[..., None, None]
        return score_R  # [n, 3, 3]

    def score_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """Calculates scaling used for scores during training."""
        return self._score_scaling[self.t_to_idx(t)]  # [n]

    def score_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Calculates weight used for scores during training."""
        return self.score_scaling(t) ** 2  # [n]

    # @custom_fwd(cast_inputs=False)
    # @custom_bwd
    def forward_marginal(
        self, R_0: torch.Tensor, t: torch.Tensor, sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples from the forward diffusion process at time index t.

        Args:
            R_0: [n, 3, 3]
            t: [n]

        Returns:
            R_t: [n, 3, 3]
            rot_score: [n, 3, 3]
        """

        sampled_R = self.sample(t).to(R_0.device)
        R_t = torch.einsum("...ij,...jk->...ik", sampled_R, R_0)
        R_0_ = sampled_R if sample else R_0  # as IGSO3 commutes
        score_R = self.score(R_t, R_0_, t, sample=sample)
        return R_t, score_R

    def reverse(
        self,
        R_t: torch.Tensor,
        score_t: torch.Tensor,
        t: float | torch.Tensor,
        dt: float | torch.Tensor,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Simulates the reverse SDE for 1 step using a Geodesic random walk.
        We assume that t corresponds to the whole batch.

        Args:
            R_t: [n, 3, 3]
            score_t: [n, 3, 3]
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            noise_scale: scaling factor for noise.

        Returns:
            R_t_1: [n, 3, 3] rotation matrix at next step.
        """
        t_ = torch.tensor([t], device=R_t.device) if not isinstance(t, torch.Tensor) else t.view(-1)  # [1] or [n]
        dt_ = torch.tensor([dt], device=R_t.device) if not isinstance(dt, torch.Tensor) else dt.view(-1)
        g_t = self.diffusion_coef(t_)
        # Get dt, g_t to same dimensions
        dt_ = dt_.view(-1, *([1] * (score_t.ndim - 1)))
        g_t = g_t.view(-1, *([1] * (score_t.ndim - 1)))
        z = so3_utils.tangent_gaussian(R_t)  # NOTE: the multiplication of R_t is redundant
        perturb = (g_t**2) * score_t * dt_ + noise_scale * g_t * torch.sqrt(dt_) * z  # [num_batch, num_frags, 3, 3]
        R_t_1 = so3_utils.expmap(R_t, perturb)  # [n, 3, 3]
        return R_t_1


if __name__ == "__main__":
    so3_diffuser = SO3Diffuser(
        schedule="logarithmic",
        min_sigma=0.01,
        max_sigma=1.5,
        num_sigma=1000,
        num_omega=2000,
        cache_path=Path("/homes/lezhang/toy-dock/runs"),
    )

    n = 5

    print("\ntest sample_igso3_angle")
    t = torch.rand(n)
    angles = so3_diffuser.sample_igso3_angle(t)
    print("angles:", angles.shape)

    print("\ntest sample")
    sampled_R = so3_diffuser.sample(t)
    print("sampled_R:", sampled_R.shape)

    print("\ntest score")
    t = torch.rand(n)
    R_id = torch.eye(3).unsqueeze(0).expand(n, 3, 3)
    R_t, score_t = so3_diffuser.forward_marginal(R_id, t)
    print("R_t, score_t:", R_t.shape, score_t.shape)

    print("\ntest reverse")
    t = t[0].item()
    dt = 0.1
    R_t_1 = so3_diffuser.reverse(R_t, score_t, t, dt)
    print("R_t_1:", R_t_1.shape)

    print("\ntest log")
    print("Log(R_t_1)[0]:", so3_utils.Log(R_t_1)[0])
    R_prime = so3_utils.exp(so3_utils.log(R_t_1))
    print("R_prime:", R_prime[0])
    print("R_t_1[0]:", R_t_1[0])
    print("norm(R_prime[0]-R_t_1[0]):", torch.linalg.norm(R_prime[0] - R_t_1[0]))
    print(torch.allclose(R_prime[0], R_t_1[0]))

    print("\ntest so3_utils for masking by zero")
    R_ = torch.zeros([n, 3, 3])
    print("Log(R_)[0]:", so3_utils.Log(R_)[0])
    t = torch.rand(n)
    print("score(R_|R_, t)[0]:", so3_diffuser.score(R_, R_, t)[0])
    print("rot_from_mat(R_):", so3_utils.rotation_vector_from_matrix(R_.reshape(-1, 3, 3)))
    print("regularize(zeros):", so3_utils.regularize(torch.zeros([n, 3])))
