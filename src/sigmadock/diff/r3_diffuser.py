import torch

"""Heavily adapted from https://github.com/jasonkyuyim/se3_diffusion."""


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(
        self,
        min_beta: float,
        max_beta: float,
    ) -> None:
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self.min_beta = min_beta
        self.max_beta = max_beta

    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        # t: [n]
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        return self.min_beta + t * (self.max_beta - self.min_beta)

    def diffusion_coef(self, t: torch.Tensor) -> torch.Tensor:
        """Time-dependent diffusion coefficient for self.reverse."""
        return torch.sqrt(self.beta_t(t))

    def drift_coef(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Time-dependent drift coefficient for self.reverse."""
        # x: [n, 3]
        return -self.beta_t(t)[:, None] * x / 2

    def sample_ref(self, n: int, device: str) -> torch.Tensor:
        return torch.randn(n, 3, device=device)

    def integrated_beta_t(self, t: torch.Tensor) -> torch.Tensor:
        integrated_beta = t * self.min_beta + (1 / 2) * (t**2) * (self.max_beta - self.min_beta)
        return integrated_beta

    def calc_trans_0(self, score_t: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Estimates E(x_0|x_t) from the score."""
        # score_t, x_t: [n, 3], t: [n]
        alpha_t = torch.exp(-self.integrated_beta_t(t) / 2)
        var_t = 1 - alpha_t**2
        x_0_given_xt = (x_t + var_t[:, None] * score_t) / alpha_t[:, None]
        return x_0_given_xt

    def score(self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Computes the conditional score of x_t|x_0 at time t."""
        # x_t, x_0: [n, 3], t: [n]
        alpha_t = torch.exp(-self.integrated_beta_t(t) / 2)
        var_t = 1 - alpha_t**2
        conditional_score = -(x_t - alpha_t[:, None] * x_0) / var_t[:, None]
        return conditional_score

    def forward_marginal(self, x_0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples marginal p(x_t | x_0).

        Args:
            x_0: [n, 3] initial positions in Angstroms.
            t: [n] in [0, 1].

        Returns:
            x_t: [n, 3] positions at time t in Angstroms.
            score_t: [n, 3] score at time t in scaled Angstroms.
        """
        z = torch.randn_like(x_0)
        alpha_t = torch.exp(-self.integrated_beta_t(t) / 2)
        x_t = alpha_t[:, None] * x_0 + torch.sqrt(1 - alpha_t**2)[:, None] * z
        score_t = self.score(x_t, x_0, t)
        return x_t, score_t

    def conditional_var(self, t: torch.Tensor) -> torch.Tensor:
        """Conditional variance of p(xt|x0): sigma(t) = 1 - alpha_t^2"""
        return 1 - torch.exp(-self.integrated_beta_t(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Extract conditional sigma(t) of p(xt|x0)"""
        return torch.sqrt(self.conditional_var(t))  # [n]

    def score_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """Scaling of the scores. Equivalent to 1/sigma(t)"""
        return 1 / torch.sqrt(self.conditional_var(t))  # [n]

    def input_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """Scaling of the inputs -> s(t) == alpha(t)"""
        return 1 / torch.sqrt(torch.exp(self.integrated_beta_t(t)))

    def score_weight(self, t: torch.Tensor) -> torch.Tensor:
        """Weight for the score function."""
        return self.sigma(t) ** 2  # [n]

    def reverse(
        self, x_t: torch.Tensor, score_t: torch.Tensor, t: float, dt: float, noise_scale: float = 1.0
    ) -> torch.Tensor:
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [n, 3] current positions at time t in angstroms.
            score_t: [n, 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            noise_scale: scaling factor for noise.

        Returns:
            x_t_1: [num_batch, num_frags, 3] positions at next step t-1.
        """
        t_ = torch.tensor([t], device=x_t.device) if not isinstance(t, torch.Tensor) else t.view(-1)  # [1] or [n]
        g_t = self.diffusion_coef(t_)  # []
        f_t = self.drift_coef(x_t, t_)
        z = torch.randn_like(x_t)
        # [n, 3]
        perturb = (f_t - g_t**2 * score_t) * dt + noise_scale * g_t * torch.sqrt(dt) * z
        x_t_1 = x_t - perturb
        return x_t_1


if __name__ == "__main__":
    r3_diffuser = R3Diffuser(min_beta=0.1, max_beta=20)

    n = 5

    x_0 = torch.randn(n, 3)
    t = torch.rand(n)

    print("testing forward_marginal")
    x_t, score_t = r3_diffuser.forward_marginal(x_0, t)
    print("x_t, score_t:", x_t.shape, score_t.shape)

    print("\ntesting reverse")
    dt = 0.1
    t = t[0].item()
    x_t_1 = r3_diffuser.reverse(x_t, score_t, t, dt)
    print("x_t_1:", x_t_1.shape)
