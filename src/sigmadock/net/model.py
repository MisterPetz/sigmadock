import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

from sigmadock.chem import RESIDUE_MAP
from sigmadock.net.edge_rot_mat import init_edge_rot_mat
from sigmadock.net.encoder import (
    AtomDiffusionEncoder,
    ChemistryEdgeEncoder,
    EdgeMixer,
    LigandVirtualDeepEncoder,
    LigandVirtualEncoder,
    ProteinResidueEncoder,
    RadialEmbeddingBlock,
)
from sigmadock.net.input_block import EdgeDegreeEmbedding
from sigmadock.net.layer_norm import get_normalization_layer
from sigmadock.net.module_list import ModuleListInfo
from sigmadock.net.radial_function import RadialFunction
from sigmadock.net.smearing import get_smearing
from sigmadock.net.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from sigmadock.net.timestep_embedder import get_timestep_embedding
from sigmadock.net.transformer_block import (
    SO2EquivariantGraphAttention,
    TransBlockV2,
)
from sigmadock.oracle import HPARAMS, HParams


class EquiformerV2(nn.Module):
    def __init__(  # noqa: C901
        self,
        # Node-Edge Encoding parameters
        atom_feature_dims: list[int],  # Chemistry-based node features
        edge_feature_dims: list[int],  # Chemistry-based edge features
        average_degrees: list[int],
        use_esm_embeddings: bool = True,
        # NOTE changed rel_distance to default True - this is the best setting.
        rel_distance: bool = True,
        zero_init_last: bool = True,
        use_edge_mixer: bool = False,
        # Interactions
        protein_ligand_interactions: bool = True,
        ligand_ligand_interactions: bool = True,
        # ------------ Architecture Dimensional Parameters -------------
        # Base Module Block Parameters
        num_layers: int = 6,
        num_heads: int = 4,
        # Feature dimensions (node, edges)
        sphere_channels: int = 256,
        edge_channels: int = 64,
        # Distance Smearing
        smearing_type: str = "fourier",  # "gaussian", "sigmoid", "silu", "fourier"
        radial_cutoff_function: str = "bessel",  # "gaussian", "bessel"
        distance_expansion_dim: int = 32,
        num_polynomial_cutoff: int = 8,
        residue_emb_dim: int = 16,
        # Time embedding
        t_emb_dim: int = 32,
        t_emb_type: str = "sinusoidal",
        time_in_edges: bool = False,
        t_emb_scale: float = 10000.0,
        # Hidden layers & linear weight sizing
        attn_hidden_channels: int = 64,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 256,
        norm_type: str = "rms_norm_sh",
        lmax_list: list[int] | None = None,
        mmax_list: list[int] | None = None,
        # Whether to use different edge MLPs for different edge types
        share_edge_mlp: bool = False,
        # -------------------- Default Parameters ----------------------
        grid_resolution: Optional[int] = None,
        use_m_share_rad: bool = False,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = "normal",
        hparams: HParams | None = None,
        **kwargs: dict,
        # NOTE: kwargs are for future compatibility, do not use them in the model.
    ) -> None:
        if lmax_list is None:
            lmax_list = [3]
        if mmax_list is None:
            mmax_list = [2]
        super().__init__()
        # INIT Construction hparams from oracle if not provided
        if hparams is None:
            print("[INFO] No HPARAMS provided, using default HPARAMS from oracle.py")
            self.hparams = HPARAMS
        else:
            print("[WARNING] Using provided HPARAMS that could be modified from oracle.py")
            self.hparams = hparams

        # Check unused kwargs
        ignored_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if len(ignored_kwargs) > 0:
            print(f"[WARN] Ignored kwargs in EquiformerV2: {ignored_kwargs}. Please check for typos unless unintended.")

        # old parameters - NOTE: check if we still need all of this
        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.use_m_share_rad = use_m_share_rad

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init

        # Encoding Params
        self.num_edge_types = self.hparams.num_edge_entities
        self.share_edge_mlp = share_edge_mlp
        self.protein_ligand_interactions = protein_ligand_interactions
        self.ligand_ligand_interactions = ligand_ligand_interactions

        # ESM3 Embeddings on protein atoms.
        self.use_esm_embeddings = use_esm_embeddings

        # Rel distance makes boundary edges contain relataive distance at current time (True is best)
        self.rel_distance = rel_distance
        self.distance_smearing = get_smearing(smearing_type)
        self.radial_cutoff_function = radial_cutoff_function
        self.average_degrees = torch.tensor(average_degrees)

        # Distance Expansion: Global and Local
        self.distance_expansion_dim = distance_expansion_dim
        self.num_polynomial_cutoff = num_polynomial_cutoff

        self.t_emb_dim = t_emb_dim
        self.time_in_edges = time_in_edges
        self.atom_feature_dims = atom_feature_dims  # NOTE: should make this into a hyper-parameter??
        self.edge_channels = edge_channels
        self.edge_feature_dims = edge_feature_dims
        self.use_edge_mixer = use_edge_mixer

        self.t_emb_type = t_emb_type
        self.t_emb_scale = t_emb_scale

        # Initialise irreps resolution
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels

        # Get time embedding function
        self.time_embedder = get_timestep_embedding(t_emb_type, t_emb_dim, t_emb_scale)
        # Define additional feature dimensions for each protein node type
        self.residue_embedder = ProteinResidueEncoder(emb_dim=residue_emb_dim)

        # ----------- Node featurisation ----------------
        self.node_encoders = nn.ModuleDict()

        # Additional Protein Features
        protein_atom_additional_feats = [residue_emb_dim, self.hparams.esm.pos_dim]
        protein_virtual_addditional_feats = []
        if self.use_esm_embeddings:
            protein_virtual_addditional_feats.append(self.hparams.esm.embedding_dim)

        # Loop through all defined node types from HPARAMS
        for node_name, _ in self.hparams.node_entity.entity_indices.items():
            if node_name in ["ligand_atom", "ligand_anchor", "ligand_dummy"]:
                encoder = AtomDiffusionEncoder(
                    emb_dim=self.sphere_channels_all,
                    t_emb_dim=t_emb_dim,
                    categorical_features=atom_feature_dims,
                    linear_aggregate=False,
                )
            elif node_name == "protein_atom":
                # Define additional features for protein atoms
                encoder = AtomDiffusionEncoder(
                    emb_dim=self.sphere_channels_all,
                    t_emb_dim=t_emb_dim,
                    categorical_features=atom_feature_dims,
                    additional_features=protein_atom_additional_feats,
                    linear_aggregate=False,
                )
            elif node_name == "ligand_virtual":
                encoder = LigandVirtualEncoder(
                    emb_dim=self.sphere_channels_all,
                    t_emb_dim=t_emb_dim,
                )
                # Lig Virtual Encoder is an aggregator of connected ligand nodes
                encoder = LigandVirtualDeepEncoder(
                    input_dim=self.sphere_channels_all,
                    output_dim=self.sphere_channels_all,
                    t_emb_dim=t_emb_dim,
                )
            elif node_name == "protein_virtual":
                encoder = AtomDiffusionEncoder(
                    emb_dim=self.sphere_channels_all,
                    t_emb_dim=t_emb_dim,
                    categorical_features=[len(RESIDUE_MAP)],  # 20 standard residue types + 1 UNK
                    additional_features=protein_virtual_addditional_feats,
                    linear_aggregate=False,
                )
            else:
                raise ValueError(f"Unknown node type '{node_name}' in HPARAMS.")

            self.node_encoders[node_name] = encoder

        # ----------- Edge featurisation ----------------
        # Create a single ModuleDict for ALL chemistry-based edge encoders, keyed by edge name
        self.chemistry_encoders = nn.ModuleDict()
        self.edge_channels = edge_channels
        self.edge_channels_list = [
            distance_expansion_dim + edge_channels,
            edge_channels,
            edge_channels,
        ]

        # Get the set of indices that are supposed to have a cutoff for verification
        cutoff_indices = self.hparams.get_edge_group_indices("has_cutoff")
        # Add chemistry edges
        chemistry_edge_names = self.hparams.edge_entity.entity_groups["chemistry"]

        # Add other non-chemical edges (virtual-virtual, interaction, etc.)
        for edge_name in chemistry_edge_names:
            # Use the detailed ChemistryEdgeEncoder for these types
            encoder = ChemistryEdgeEncoder(
                edge_channels,
                edge_feature_dims,
                linear_aggregate=False,
            )
            self.chemistry_encoders[edge_name] = encoder

        # ------------ Edge distance expansions -------------
        # Create a single ModuleDict for all distance encoders
        self.distance_encoders = nn.ModuleDict()

        for edge_name, edge_spec in self.hparams.get_edge_specs(
            list(self.hparams.edge_specs.keys()), use_scaling=True
        ).items():
            # Get the index of this edge type
            edge_idx = self.hparams.edge_entity.entity_indices[edge_name]

            # Skip edge types based on interaction settings
            if edge_name == "inter_complex" and not self.protein_ligand_interactions:
                print("[INFO] DIST | Skipping inter_complex edges as protein_ligand_interactions is False")
                continue
            if edge_name == "inter_fragments" and not self.ligand_ligand_interactions:
                print("[INFO] DIST | Skipping inter_fragments edges as ligand_ligand_interactions is False")
                continue

            # Check if this edge type uses a cutoff
            if edge_spec.r_max is not None:
                # Ensure that any edge with an r_max is correctly categorized.
                assert edge_idx in cutoff_indices, (
                    f"Configuration Error: Edge '{edge_name}' (id: {edge_idx}) is defined with 'r_max' "
                    f"but is not in the 'has_cutoff' group in HPARAMS."
                )
                encoder = RadialEmbeddingBlock(
                    r_max=edge_spec.r_max,
                    num_bessel=distance_expansion_dim + edge_channels,
                    num_polynomial_cutoff=self.num_polynomial_cutoff,
                    radial_type=self.radial_cutoff_function,
                )
            else:
                # Note only chemistry edges get extra edge channels for features (rest don't rlly have any features anyway)  # noqa: E501
                encoder = self.distance_smearing(
                    start=edge_spec.start,
                    stop=edge_spec.stop,
                    num_basis=distance_expansion_dim
                    if edge_name in chemistry_edge_names
                    else distance_expansion_dim + edge_channels,
                    basis_width_scalar=edge_spec.basis_width_scalar,
                )
            self.distance_encoders[edge_name] = encoder

        # ------------ Edge Feature Mixer (Feature, Distance Expansion, Time (Optionally)) -------------
        if self.use_edge_mixer:
            self.edge_mixers = nn.ModuleDict()
            mixer_in_dim = self.edge_channels + self.distance_expansion_dim
            if self.time_in_edges:
                mixer_in_dim += self.t_emb_dim

            # The output dim should match what the TransBlockV2 expects
            mixer_out_dim = self.edge_channels_list[0]

            for edge_name in self.hparams.edge_entity.entity_indices:
                if edge_name == "inter_complex" and not self.protein_ligand_interactions:
                    continue
                if edge_name == "inter_fragments" and not self.ligand_ligand_interactions:
                    continue
                self.edge_mixers[edge_name] = EdgeMixer(in_dim=mixer_in_dim, out_dim=mixer_out_dim)
        else:
            if self.time_in_edges:
                raise NotImplementedError(
                    "Edge mixing is disabled, but time embedding in edges is enabled. "
                    "Please set 'use_edge_mixer' to True or disable 'time_in_edges'."
                )

        # Initialise the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialise conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialise the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(f"({max(self.lmax_list)}, {max(self.lmax_list)})")
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.edge_channels_list,
            self.average_degrees,
            self.num_edge_types,
            self.share_edge_mlp,
        )

        # Initialize the blocks for each layer
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.edge_channels_list,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
                self.num_edge_types,
                self.share_edge_mlp,
            )
            self.blocks.append(block)

        # Output blocks for outputing forces
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels,
        )

        self.force_block = SO2EquivariantGraphAttention(
            self.sphere_channels,
            self.attn_hidden_channels,
            self.num_heads,
            self.attn_alpha_channels,
            self.attn_value_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.SO3_grid,
            self.edge_channels_list,
            self.use_m_share_rad,
            self.attn_activation,
            self.use_s2_act_attn,
            self.use_attn_renorm,
            self.use_gate_act,
            self.use_sep_s2_act,
            alpha_drop=0.0,
            num_edge_types=self.num_edge_types,
            share_edge_mlp=self.share_edge_mlp,
        )

        # Initialize weights
        self.apply(partial(self._init_weights, zero_init_last=zero_init_last))
        self.apply(self._uniform_init_rad_func_linear_weights)

    def generate_geometry(self, data: Data, pos_key: str = "pos_t") -> tuple[torch.Tensor, torch.Tensor]:
        assert pos_key in data, f"pos_key {pos_key} not found in data"
        j, i = data.edge_index
        edge_distance_vec = data.get(pos_key)[j] - data.get(pos_key)[i]
        edge_distance = edge_distance_vec.norm(dim=-1)
        return edge_distance, edge_distance_vec

    def compute_node_embedding(self, data: Data, t: torch.Tensor) -> torch.Tensor:  # noqa: C901
        # 1. Compute and broadcast time embeddings
        repeats = torch.bincount(data.batch).long()
        t_emb = self.time_embedder(t)
        t_emb = torch.repeat_interleave(t_emb, repeats, dim=0)

        # 2. Initialize a tensor to hold the L=0 node features
        total_num_nodes = len(data.x)
        node_features_l0 = torch.zeros(
            total_num_nodes,
            self.sphere_channels_all,
            device=data.x.device,
            dtype=data.x.dtype,
        )

        # 3. Create a reverse mapping and pre-calculate protein masks
        idx_to_name = {idx: name for name, idx in self.hparams.node_entity.entity_indices.items()}
        name_to_idx = {
            name: idx for idx, name in self.hparams.edge_entity.entity_indices.items()
        }
        protein_indices = self.hparams.get_node_group_indices("protein")
        is_protein = torch.isin(data.node_entity, torch.tensor(protein_indices, device=data.x.device))
        protein_node_entities = data.node_entity[is_protein]
        is_protein_atom = protein_node_entities == self.hparams.get_node_idx("protein_atom")
        is_protein_virtual = protein_node_entities == self.hparams.get_node_idx("protein_virtual")

        # 3.1 Compute residue embeddings for protein atoms
        protein_residue_types = data.residue_types[is_protein]
        protein_residue_emb = self.residue_embedder(protein_residue_types)

        # ---- STAGE 1, COMPUTE REAL NODE FEATURES ----
        node_names = {idx_to_name[idx.item()] for idx in data.node_entity.unique()}
        virtuals = {a for a in node_names if "virtual" in a}
        reals = node_names - virtuals

        for node_name in reals:
            node_idx = self.hparams.node_entity.entity_indices.get(node_name)
            if node_idx is None:
                continue

            node_mask = data.node_entity == node_idx
            if not node_mask.any():
                continue

            encoder = self.node_encoders[node_name]

            if node_name == "protein_atom":
                additional_features = [
                    protein_residue_emb[is_protein_atom],
                    data.protein_embeddings["positional_embeddings"][is_protein_atom],
                ]
                features_in = torch.cat([data.x[node_mask], *additional_features], dim=1)
                node_features = encoder(x=features_in, time_features=t_emb[node_mask])
                node_features_l0[node_mask] = node_features
            else:  # ligand_atom, ligand_anchor, ligand_dummy
                node_features = encoder(x=data.x[node_mask], time_features=t_emb[node_mask])
                node_features_l0[node_mask] = node_features

        # ---- STAGE 2, COMPUTE VIRTUAL NODE FEATURES ----
        for node_name in virtuals:
            node_idx = self.hparams.node_entity.entity_indices.get(node_name)
            if node_idx is None:
                continue

            node_mask = data.node_entity == node_idx
            if not node_mask.any():
                continue

            encoder = self.node_encoders[node_name]

            if node_name == "ligand_virtual":
                # Context-Aware Logic for Ligand Virtual Nodes
                v2a_edge_mask = data.edge_entity == self.hparams.get_edge_idx("ligand_v2a")
                v2a_edges = data.edge_index[:, v2a_edge_mask]

                # We must filter the edges to ensure we only aggregate from real atoms TO virtual nodes, not the other way around.  # noqa: E501
                dest_is_virtual = data.node_entity[v2a_edges[0]] == self.hparams.get_node_idx("ligand_virtual")
                src_is_real = data.node_entity[v2a_edges[1]] != self.hparams.get_node_idx("ligand_virtual")

                valid_edges = v2a_edges[:, dest_is_virtual & src_is_real]

                virtual_nodes, real_atom_neighbors = valid_edges[0], valid_edges[1]
                neighbor_features = node_features_l0[real_atom_neighbors]
                aggregated_feats = scatter(
                    neighbor_features,
                    virtual_nodes,
                    dim=0,
                    dim_size=total_num_nodes,
                    reduce="mean",
                )

                contextual_feats = aggregated_feats[node_mask]
                node_features = encoder(contextual_feats, time_features=t_emb[node_mask])
                node_features_l0[node_mask] = node_features

            elif node_name == "protein_virtual":
                # Protein virtual nodes use their residue type, as before
                categorical_in = protein_residue_types[is_protein_virtual].unsqueeze(-1)
                additional_features = []
                if self.use_esm_embeddings:
                    additional_features.append(data.protein_embeddings["esm_embeddings"][is_protein_virtual])
                features_in = torch.cat([categorical_in, *additional_features], dim=1)
                node_features = encoder(x=features_in, time_features=t_emb[node_mask])
                node_features_l0[node_mask] = node_features

        # Initialize the final SO3_Embedding tensor
        x_embedding = SO3_Embedding(
            len(data.x),
            self.lmax_list,
            self.sphere_channels,
            device=data.x.device,
            dtype=data.x.dtype,
        )
        # Place the computed features into the SO3_Embedding tensor
        offset = 0
        offset_res = 0
        for i in range(self.num_resolutions):
            # For each resolution, select the appropriate slice from the L=0 features
            sphere_channels_for_res = self.sphere_channels
            features_for_res = node_features_l0[:, offset : offset + sphere_channels_for_res]

            # Place these features into the l=0, m=0 component for this resolution
            x_embedding.embedding[:, offset_res, :] = features_for_res

            # Update offsets for the next resolution
            offset += sphere_channels_for_res
            offset_res += int((self.lmax_list[i] + 1) ** 2)

        return x_embedding

    def compute_edge_embedding(
        self,
        data: Data,
        edge_distance: torch.Tensor,
        # Note these are both normalized distances, not absolute positions!
        ref_pos_key: str = "pos_0",
        cur_pos_key: str = "pos_t",
        rel_distance: bool = True,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes edge embeddings by combining chemistry and distance features.
        This version is refactored to use unified ModuleDicts for clarity and robustness.

        Args:
            data (Data): Input data containing edge attributes and indices.
            edge_distance (torch.Tensor): Precomputed edge distances.
            ref_pos_key (str): Key for reference positions in the data.
            cur_pos_key (str): Key for current positions in the data.
            rel_distance (bool): Whether to use relative distances for boundary edges.
        Returns:
            torch.Tensor: Edge embeddings combining chemistry and distance features.
        """
        # Extract edge features from edge types
        total_num_edges = len(data.edge_entity)

        # 1. Initialize output tensors for chemistry and distance features.
        final_edge_embedding = torch.zeros(
            [total_num_edges, self.edge_channels + self.distance_expansion_dim],
            device=data.x.device,
        )

        # 2. Pre-fetch HPARAMS for efficiency - avoids repeated lookups inside the loop.
        boundary_indices = self.hparams.get_edge_group_indices("boundary")
        chemistry_indices = self.hparams.get_edge_group_indices("chemistry")
        idx_to_name = {idx: name for name, idx in self.hparams.edge_entity.entity_indices.items()}

        # 3. Loop through each unique edge type present in the current batch.
        # This is more efficient than looping through all possible edge types.
        for edge_type_idx in data.edge_entity.unique():
            edge_type_idx_int = edge_type_idx.item()
            edge_name = idx_to_name[edge_type_idx_int]
            edge_mask = data.edge_entity == edge_type_idx

            # Look up the correct distance encoder by name.
            dist_encoder = self.distance_encoders[edge_name]

            # Chemistry + Distance edges
            if edge_type_idx in chemistry_indices:
                # Look up the correct encoder by name from the unified ModuleDict.
                chem_encoder = self.chemistry_encoders[edge_name]
                # For edges with chemical properties, pass the edge attributes.
                edge_feats = chem_encoder(data.edge_attr[edge_mask].long())
                dist_feats = dist_encoder(edge_distance[edge_mask])
                # Combine chemistry and distance features for these edges
                final_edge_embedding[edge_mask] = torch.cat([edge_feats, dist_feats], dim=1)

            # Distance Edges (No Chemistry or Additional Features)
            # NOTE we assume these edges do not have edge attributes!
            else:
                # The logic for "boundary" edges is special: it uses relative distance.
                if edge_type_idx_int in boundary_indices and rel_distance:
                    j0, i0 = data.edge_index[:, edge_mask]
                    original_edge_distance = (data.get(ref_pos_key)[j0] - data.get(ref_pos_key)[i0]).norm(dim=-1)
                    current_edge_distance = (data.get(cur_pos_key)[j0] - data.get(cur_pos_key)[i0]).norm(dim=-1)
                    distance_input = torch.abs(current_edge_distance - original_edge_distance)
                else:
                    # All other edge types use the pre-computed absolute distance.
                    distance_input = edge_distance[edge_mask]
                final_edge_embedding[edge_mask] = dist_encoder(distance_input)

        # Optionally add time embedding
        if self.use_edge_mixer:
            if self.time_in_edges:
                assert t_emb is not None, "Time embedding must be provided if time_in_edges is True."
                edge_batch_idx = data.batch[data.edge_index[0]]
                edge_t_emb = t_emb[edge_batch_idx]
                combined_feats = torch.cat([final_edge_embedding, edge_t_emb], dim=1)

            # --- Apply the correct mixer for each edge type ---
            for edge_type_idx in data.edge_entity.unique():
                edge_name = idx_to_name[edge_type_idx.item()]
                edge_mask = data.edge_entity == edge_type_idx

                # Select the specialized mixer for this edge type
                mixer = self.edge_mixers[edge_name]

                # Mix the features and place them in the final output tensor
                final_edge_embedding[edge_mask] = mixer(combined_feats[edge_mask])

        return final_edge_embedding

    def forward(self, data: Batch, t: torch.Tensor, **kwargs: dict) -> None:
        # NOTE only operates with Batch objects not supported. Use Batch object instead.")
        if not isinstance(data, Batch):
            raise ValueError("Input data must be a Batch object.")

        # Prune edges that do not have valid mask node!
        keep_edge = data.mask[data.edge_index[0]] & data.mask[data.edge_index[1]]
        num_pruned: int = torch.sum(~keep_edge).item()
        data.edge_index = data.edge_index[:, keep_edge]
        data.edge_entity = data.edge_entity[keep_edge]
        data.edge_attr = data.edge_attr[keep_edge]

        self.batch_size = data.num_graphs

        # Generate edge geometry
        edge_distance, edge_distance_vec = self.generate_geometry(data, pos_key="pos_t")

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = init_edge_rot_mat(edge_distance_vec)

        # Initialise the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        # Initialize node features
        x = self.compute_node_embedding(data, t)

        # Compute edge and edge degree embedding
        edge_embedding = self.compute_edge_embedding(
            data,
            edge_distance,
            cur_pos_key="pos_t",
            ref_pos_key="pos_0",
            rel_distance=self.rel_distance,
        )
        edge_degree = self.edge_degree_embedding(data.node_entity, edge_embedding, data.edge_index, data.edge_entity)
        x.embedding = x.embedding + edge_degree.embedding

        # Apply EquiformerV2 blocks
        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                edge_embedding,
                data.edge_index,
                data.edge_entity,
                batch=data.batch,  # for GraphDropPath
            )
            # TODO: potentially add time embeddings layer-wise: Potentially better inductive bias to preserve time.
            # NOTE: similar to DiT.

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        # Compute forces
        forces = self.force_block(x, edge_embedding, data.edge_index, data.edge_entity)
        # Narrow Opt is gathering the L=1 outputs here.
        forces = forces.embedding.narrow(1, 1, 3)
        forces = forces.view(-1, 3)
        return forces

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m: nn.Module, zero_init_last: bool = True) -> None:
        if isinstance(m, (torch.nn.Linear, SO3_LinearV2)):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)
            if (m.out_features == 1) and zero_init_last:
                # Zero-weight init to Force Block Last layer -> Keep output var == 1
                assert len(list(m.named_children())) == 0
                torch.nn.init.constant_(m.weight, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0) if m.bias is not None else None
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m) -> None:  # noqa
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m) -> None:  # noqa
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    """NOTE: Do we need no_weight_decay? - See original implementations"""


if __name__ == "__main__":
    model = EquiformerV2(num_layers=1)
