"""Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from typing import Any

class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_basis: int = 50,
        basis_width_scalar: float | None = None,
        **ignored_kwargs: Any,
    ) -> None:
        super(GaussianSmearing, self).__init__()
        if basis_width_scalar is None:
            basis_width_scalar = (stop - start) / 10
        self.num_basis = num_basis
        offset = torch.linspace(start, stop, num_basis)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class SigmoidSmearing(nn.Module):
    """
    Projects scalar distances into sigmoid bump features.
    Sigmoids are centered on mu_k, with steepness gamma *per center*,
    so changing start/stop doesn't alter the shape.
    """
    def __init__(self,
                 start: float = 0.0,
                 stop:  float = 10.0,
                 num_basis: int = 32,
                 gamma: float = 1.0,
                 learn_centers: bool = False,
                 **ignored_kwargs: Any,
                 ) -> None:
        super().__init__()
        # initialize centers
        mus = torch.linspace(start, stop, num_basis)
        self.centers = nn.Parameter(mus, requires_grad=learn_centers)

        # compute center‐to‐center spacing
        # (if num_centers=1, spacing=1 to avoid div0)
        self.spacing = (stop - start) / max(num_basis - 1, 1)
        self.stop = stop  # store for reference
        # gamma is now "per‐center" steepness
        self.gamma = gamma

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # (...,) → (...,1)
        if dist.dim() == 1 or dist.shape[-1] != 1:
            dist = dist.unsqueeze(-1)

        # normalize offset by spacing, so 1 unit = 1 center
        diff = (dist - self.centers) / self.spacing  # (..., num_centers)

        # now gamma controls steepness in *centers* not Ångstroms
        return torch.sigmoid(self.gamma * diff)


class SiLUSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_basis: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super(SiLUSmearing, self).__init__()
        self.num_basis = num_basis
        self.fc1 = nn.Linear(2, num_basis)
        self.act = nn.SiLU()

    def forward(self, dist):
        x_dist = dist.view(-1, 1)
        x_dist = torch.cat([x_dist, torch.ones_like(x_dist)], dim=1)
        x_dist = self.act(self.fc1(x_dist))
        return x_dist
    

class TotalFourierSmearing(nn.Module):
    """
    Projects a scalar distance tensor into a concatenated [sin, cos] Fourier basis
    with an exponential decay envelope.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_basis: int = 50,
        decay_rate: float = 2.0 ** 0.5,
        **ignored_kwargs: Any,
    ) -> None:
        """
        Args:
            start (float): Minimum distance (unused internally here).
            stop (float): Maximum distance (cutoff), used to normalize.
            num_basis (int): Number of distinct frequencies (will produce 2*num_basis features).
            decay_rate (float): Exponential envelope decay constant.
        """
        super().__init__()
        self.decay_rate: float = decay_rate
        self.d_max: float = stop
        # Create a fixed Parameter holding the frequency multipliers
        freqs: torch.Tensor = torch.linspace(0.5, 8.0, num_basis // 2)
        # self.frequencies: nn.Parameter = nn.Parameter(freqs, requires_grad=False)
        self.register_buffer("frequencies", freqs)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist (Tensor): Tensor of shape (...,) or (...,1) containing pairwise distances.

        Returns:
            Tensor of shape (..., 2*num_basis), with [sin, cos] channels attenuated
            by the exponential envelope.
        """
        # ensure last-dim is singleton so broadcasting works
        if dist.ndim == dist.unsqueeze(-1).ndim - 1:
            dist = dist.unsqueeze(-1)  # (..., 1)

        # Exponential decay envelope: shape (...,1)
        envelope: torch.Tensor = torch.exp(-self.decay_rate * dist / self.d_max)

        # Argument for sine/cosine: shape (..., num_basis)
        arg: torch.Tensor = self.frequencies * torch.pi * dist / self.d_max

        sin_feat: torch.Tensor = torch.sin(arg)
        cos_feat: torch.Tensor = torch.cos(arg)

        # Concatenate to (..., 2*num_basis) and apply envelope
        fourier_feats: torch.Tensor = torch.cat([sin_feat, cos_feat], dim=-1)
        return envelope * fourier_feats


class FourierSmearing(nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_basis: int = 50,
        decay_rate: float = 2.0 ** 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        self.start = start
        self.stop = stop
        self.d_max = stop
        self.decay_rate = decay_rate

        self.register_buffer("frequencies", torch.linspace(0.5, 8.0, num_basis))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = dist.unsqueeze(-1)  # (..., 1)
        exp_term = torch.exp(-self.decay_rate * dist / self.d_max)  # (..., 1)
        cos_term = torch.cos(self.frequencies * torch.pi * dist / self.d_max)  # (..., num_basis)
        return exp_term * cos_term  # (..., num_basis)
    
    
def get_smearing(
    smearing_type: str = "sigmoid",
) -> nn.Module:
    if smearing_type == "gaussian":
        return GaussianSmearing
    elif smearing_type == "sigmoid":
        return SigmoidSmearing
    elif smearing_type == "silu":
        return SiLUSmearing
    elif smearing_type == "symmetric-fourier":
        return FourierSmearing
    elif smearing_type == "fourier":
        return TotalFourierSmearing
    else:
        raise ValueError(f"Unknown smearing type: {smearing_type}")