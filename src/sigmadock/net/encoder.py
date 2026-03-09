import numpy as np
import torch
import torch.nn as nn

from typing import Literal

class EdgeMixer(nn.Module):
    """
    An MLP that mixes different types of edge features (chemistry, distance,
    and optionally time) into a single, rich embedding.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
        # It's good practice to initialize the mixer MLP
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class ProteinResidueEncoder(nn.Module):
    """
    Creates a learnable embedding for the 20 standard amino acid residue types.
    """
    def __init__(self, emb_dim: int, num_residue_types: int = 21):
        """
        Args:
            emb_dim (int): The dimension of the output embedding vector.
            num_residue_types (int): The number of unique residue types.
                Defaults to 21 (20 standard + 1 for 'unknown').
        """
        super().__init__()
        self.embedding = nn.Embedding(num_residue_types, emb_dim)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, residue_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residue_indices (torch.Tensor): A tensor of integer indices representing
                the residue type for each atom. Shape: [num_atoms].

        Returns:
            torch.Tensor: The learnable residue embeddings. Shape: [num_atoms, emb_dim].
        """
        return self.embedding(residue_indices)
    
    
class AtomDiffusionEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        t_emb_dim: int,
        categorical_features: list[int],
        additional_features: list[int] | None = None,
        linear_aggregate: bool = False,
    ) -> None:
        """Atom Encoder for molecular diffusion.

        Args:
            emb_dim (int): Embedding dimension
            feature_dims (list[int]): List of feature dimensions
            t_emb_dim (int): Embedding dimension for additional features
            additional_features_dim (int, optional). Defaults to 0.
            linear_aggregate (bool, optional). Defaults to False.
        """
        super().__init__()
        self.atom_embedding_list = nn.ModuleList()
        self.emb_dim = emb_dim
        self.t_emb_dim = t_emb_dim
        self.categorical_dims = categorical_features
        
        aggregate = 0
        # NOTE we add +1 for "unknown" category
        for dim in categorical_features:
            dim_plus_none = dim + 1
            emb = nn.Embedding(dim_plus_none, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
            aggregate += emb_dim

        if linear_aggregate:
            assert len(categorical_features) > 1, "Linear aggregation only works for multiple features."
            self.aggregator = nn.Sequential(
                nn.Linear(aggregate, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.aggregator = None

        self.num_categorical_features = len(categorical_features)
        self.num_additional_features = sum(additional_features) if additional_features else 0
        if self.num_additional_features > 0:
            self.additional_features_embedder = nn.Sequential(
                nn.Linear(self.num_additional_features, emb_dim),
                nn.SiLU(),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.additional_features_embedder = nn.Identity()

        self.time_projector = nn.Linear(emb_dim + t_emb_dim, emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        # x: [n_atoms, num_categorical_features + additional_features_dim]
        assert x.shape[1] == self.num_categorical_features + self.num_additional_features
        # Clone due to in-place modifications in the loop
        x_cat = x[:, : self.num_categorical_features].clone()
        x_cont = x[:, self.num_categorical_features :]
        
        # Size checks
        assert x_cat.shape[1] == self.num_categorical_features
        assert x_cont.shape[1] == self.num_additional_features
        assert time_features.shape[1] == self.t_emb_dim

        # map out-of-range values to unknown index (dim)
        for i, dim in enumerate(self.categorical_dims):
            # valid indices: 0 .. dim-1, unknown index: dim
            invalid_low = x_cat[:, i] < 0
            invalid_high = x_cat[:, i] >= dim
            x_cat[:, i] = torch.where(invalid_low | invalid_high, dim, x_cat[:, i])
            
        x_embedding = []
        if self.num_categorical_features > 0:
            x_embedding = [self.atom_embedding_list[i](x_cat[:, i].long()) for i in range(self.num_categorical_features)]
            if self.aggregator:
                x_embedding = torch.cat(x_embedding, dim=1)
                x_embedding = self.aggregator(x_embedding)
            else:
                x_embedding = torch.stack(x_embedding, dim=1)
                x_embedding = torch.sum(x_embedding, dim=1)
                # NOTE added embedding scaling.
                x_embedding = x_embedding / (self.num_categorical_features**0.5)
        else:
            x_embedding = torch.zeros((x.shape[0], self.num_categorical_features, self.emb_dim), device=x.device)

        # Add categorical features no extra feature embedding
        if self.num_additional_features > 0:
            x_embedding = x_embedding + self.additional_features_embedder(x_cont)

        # Add time embedding
        x_embedding = self.time_projector(torch.cat([x_embedding, time_features], dim=1))
        return x_embedding


class LigandVirtualDeepEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, t_emb_dim: int):
        super().__init__()
        # An MLP to process the aggregated neighbor features and time embedding
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + t_emb_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight.data)
        nn.init.xavier_uniform_(self.mlp[2].weight.data)
        # Initialize biases to zero
        nn.init.zeros_(self.mlp[0].bias.data)
        nn.init.zeros_(self.mlp[2].bias.data)

    def forward(self, aggregated_feats: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        # Concatenate the context-aware features with the time embedding
        combined_feats = torch.cat([aggregated_feats, time_features], dim=1)
        return self.mlp(combined_feats)
    
class LigandVirtualEncoder(nn.Module):
    def __init__(self, emb_dim: int, t_emb_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(1, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight.data)

        self.feature_embedder = nn.Linear(t_emb_dim + emb_dim, emb_dim)

    def forward(self, x: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        x_embedding = self.emb(torch.zeros_like(x[:, 0]).long())
        x_embedding = self.feature_embedder(torch.cat([x_embedding, time_features], dim=1))
        return x_embedding


class ChemistryEdgeEncoder(nn.Module):
    def __init__(self, edge_channels: int, feature_dims: list[int], linear_aggregate: bool = False) -> None:
        super().__init__()
        self.chemistry_embedding_list = nn.ModuleList()
        aggregate = 0
        self.feature_dims = feature_dims
        
        # NOTE we add +1 for "unknown" category
        for dim in feature_dims:
            emb = nn.Embedding(dim + 1, edge_channels)
            nn.init.xavier_uniform_(emb.weight.data)
            self.chemistry_embedding_list.append(emb)
            aggregate += edge_channels

        if linear_aggregate:
            assert len(feature_dims) > 1, "Linear aggregation only works for multiple features."
            self.aggregator = nn.Sequential(
                nn.Linear(aggregate, edge_channels),
                nn.SiLU(),
                nn.Linear(edge_channels, edge_channels),
            )
        else:
            self.aggregator = None
        self.num_edge_channels = edge_channels
        self.num_edge_features = sum(feature_dims)
        self.num_categorical_features = len(feature_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.num_categorical_features
        
        # Map out-of-range values to unknown index (dim)
        x_mod = x.clone()
        for i, dim in enumerate(self.feature_dims):
            invalid = (x_mod[:, i] < 0) | (x_mod[:, i] >= dim)
            x_mod[:, i] = torch.where(invalid, dim, x_mod[:, i])
            
        x_embedding = [self.chemistry_embedding_list[i](x_mod[:, i].long()) for i in range(self.num_categorical_features)]

        if self.aggregator:
            x_embedding = torch.cat(x_embedding, dim=1)
            x_embedding = self.aggregator(x_embedding)
        else:
            x_embedding = torch.stack(x_embedding, dim=1)
            x_embedding = torch.sum(x_embedding, dim=1)
            # Scale the embedding
            x_embedding = x_embedding / (self.num_categorical_features**0.5)
        return x_embedding

"""Adapted from https://github.com/ACEsuit/mace."""
class BesselBasis(torch.nn.Module):
    """
    Bessel basis functions
    """
    def __init__(
        self, r_max: float, num_basis: int = 8, scaling: float = 1 / 8, trainable: bool = False
    ) -> None:
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        self.scaling = scaling  # Default value, can be adjusted later
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        scaled_x = x * self.scaling
        numerator = torch.sin(self.bessel_weights * scaled_x)  # [..., num_basis]
        return self.prefactor * (numerator / (scaled_x + 1e-8))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )

class GaussianBasis(torch.nn.Module):
    """
    Gaussian basis functions
    """

    def __init__(self, r_max: float, num_basis: int = 128, trainable: bool = False) -> None:
        super().__init__()
        gaussian_weights = torch.linspace(start=0.0, end=r_max, steps=num_basis, dtype=torch.get_default_dtype())
        if trainable:
            self.gaussian_weights = torch.nn.Parameter(gaussian_weights, requires_grad=True)
        else:
            self.register_buffer("gaussian_weights", gaussian_weights)
        self.coeff = -0.5 / (r_max / (num_basis - 1)) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        x = x - self.gaussian_weights
        return torch.exp(self.coeff * torch.pow(x, 2))


class PolynomialCutoff(torch.nn.Module):
    """Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p: int = 6) -> None:
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


class RadialEmbeddingBlock(torch.nn.Module):  # for local edges and Virtual P2P edges
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: Literal["gaussian", "bessel"] = "bessel",
    ) -> None:
        super().__init__()
        if radial_type == "bessel":
            # NOTE: might be unstable for our use case
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel, trainable=True)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel, trainable=True)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(self, edge_lengths: torch.Tensor) -> torch.Tensor:  # [n_edges]
        edge_lengths = edge_lengths.unsqueeze(-1)  # [n_edges, 1]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]


if __name__ == "__main__":
    edge_encoder = RadialEmbeddingBlock(
        r_max=10.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
        radial_type="bessel",
    )

    edge_lengths = torch.rand(10)
    edge_lengths[-2] = 9.9
    edge_lengths[-1] = 10.0
    print(edge_encoder(edge_lengths))
