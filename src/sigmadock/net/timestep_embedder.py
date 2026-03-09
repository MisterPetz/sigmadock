import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    A PyTorch module that implements the standard sinusoidal time embedding
    from the DDPM paper (Ho et al., 2020), adapted for continuous time t in [0, 1],
    and followed by a learnable MLP.
    """
    def __init__(self, embedding_dim: int, max_positions: int = 10000):
        """
        Initializes the DDPMTimeEmbedding module.

        Args:
            embedding_dim (int): The dimension of the output time embedding.
            max_positions (int): Defines the scale of the time embeddings.
                A higher value creates higher-frequency features.
        """
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, but got {embedding_dim}")

        self.embedding_dim = embedding_dim
        self.max_positions = max_positions

        # A small MLP to process the sinusoidal features and make them learnable.
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Computes the time embedding for a batch of time steps.

        Args:
            timesteps (torch.Tensor): A 1D tensor of continuous time steps in the
                range [0.0, 1.0], shape [batch_size].

        Returns:
            torch.Tensor: The time embeddings, shape [batch_size, embedding_dim].
        """
        # --- Step 1: Create the Sinusoidal Basis ---
        # This logic is taken from the original DDPM paper, which expects
        # integer timesteps. To adapt it for continuous time t in [0, 1],
        # we first scale t by max_positions. This correctly applies the
        # frequency spectrum to the continuous input.
        timesteps = timesteps * self.max_positions

        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        sinusoidal_embedding = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embedding_dim % 2 == 1:  # zero pad if embedding_dim is odd
            sinusoidal_embedding = F.pad(sinusoidal_embedding, (0, 1), mode="constant")

        # --- Step 2: Project through the MLP ---
        time_embedding = self.mlp(sinusoidal_embedding)
        return time_embedding


class GaussianFourierTimeEmbedding(nn.Module):
    """
    A PyTorch module that implements Gaussian Fourier features for time/noise
    levels, as used in the Score-SDE paper (Song et al., 2021), followed
    by a learnable MLP.
    """
    def __init__(self, embedding_dim: int = 256, scale: float = 1.0):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, but got {embedding_dim}")
            
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.randn(embedding_dim // 2) * scale, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, x: torch.Tensor, return_raw: bool = False) -> torch.Tensor:
        x_proj = x[:, None] * self.W[None, :] * 2 * math.pi
        fourier_embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        if return_raw:
            return fourier_embedding
            
        time_embedding = self.mlp(fourier_embedding)
        return time_embedding
    

def get_timestep_embedding(t_emb_type: str, t_emb_dim: int, t_emb_scale: float = 128.0) -> Callable:
    if t_emb_type == "sinusoidal":
        emb_func = SinusoidalTimeEmbedding(embedding_dim=t_emb_dim, max_positions=t_emb_scale)
    elif t_emb_type == "fourier":
        emb_func = GaussianFourierTimeEmbedding(embedding_dim=t_emb_dim, scale=t_emb_scale // 16)
    else:
        raise NotImplementedError
    return emb_func
