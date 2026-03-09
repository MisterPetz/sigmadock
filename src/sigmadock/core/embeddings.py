import torch
from torch_geometric.utils import to_dense_adj


def graph_laplacian(edge_index: torch.Tensor) -> torch.Tensor:
    """Compute the Graph Laplacian L = D - A directly in PyTorch."""
    A = to_dense_adj(edge_index).squeeze(0)  # Adjacency matrix
    D = torch.diag(A.sum(dim=1))  # Degree matrix
    L = D - A  # Graph Laplacian
    return L


def graph_laplacian_SE(edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Compute L * X as a structural encoding in PyTorch."""
    L = graph_laplacian(edge_index).to(pos.dtype)
    return L @ pos  # Shape: (num_nodes, 2)


class GraphLaplacianSE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        L = graph_laplacian(edge_index).to(pos.dtype)
        return L @ pos  # Shape: (num_nodes, 2)


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, scale: int = 16) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.ger((2 * torch.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.
class ScaledFourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels: int, scale: int = 16) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.ger((2 * torch.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        # Scale by sqrt(2) to make the embedding orthonormal.
        return x * (2**0.5)
