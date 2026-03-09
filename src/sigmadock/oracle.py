from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional


# --- Parameter dataclasses ---
@dataclass(frozen=True)
class EdgeSpec:
    """
    Unified specification for edge cutoffs or distance features.
    If r_max is set, use as hard cutoff; otherwise use start/stop/basis_width_scalar for Gaussian smearing.
    """

    # Hard cutoff distance (Å)
    r_max: Optional[float] = None
    # Gaussian feature params
    start: Optional[float] = None
    stop: Optional[float] = None
    basis_width_scalar: Optional[float] = None
    scaling: Optional[float] = None  # Optional scaling factor for the edge features


@dataclass(frozen=True)
class GeneralConfig:
    # STD scale for dimensionality reduction (Å)
    dimensional_scale: float
    # Small value to avoid division by zero in diffusion process.
    epsilon_t: float = 0.01


@dataclass(frozen=True)
class ESMConfig:
    embedding_dim: int
    pos_dim: int


# --- Entity configurations ---
@dataclass(frozen=True)
class NodeEntityConfig:
    """
    Maps node entity names to integer codes and defines groups.
    """

    entity_indices: dict[str, int]
    entity_groups: dict[str, list[str]]
    # Degrees without local interactions ONLY
    global_degrees: dict[str, float]
    # Degrees with local interactions ONLY
    local_degrees: dict[str, float]


@dataclass(frozen=True)
class EdgeEntityConfig:
    """
    Maps edge entity names to integer codes and defines groups.
    """

    entity_indices: dict[str, int]
    entity_groups: dict[str, list[str]]


# --- Hyperparameters container ---
@dataclass(frozen=True)
class HParams:
    node_entity: NodeEntityConfig
    edge_entity: EdgeEntityConfig
    # Unified specs for all edge types
    edge_specs: dict[str, EdgeSpec]
    general: GeneralConfig
    esm: ESMConfig

    def __post_init__(self) -> None:
        # Ensure every defined edge entity has a spec
        missing = set(self.edge_entity.entity_indices.keys()) - set(self.edge_specs.keys())
        if missing:
            raise ValueError(f"Missing specs for edge entities: {missing}")
        # Ensure no extraneous specs not tied to an entity
        extra = set(self.edge_specs.keys()) - set(self.edge_entity.entity_indices.keys())
        if extra:
            raise ValueError(f"Edge specs defined for unknown entities: {extra}")

    @property
    def num_node_entities(self) -> int:
        """
        Returns number of unique node entity codes (max index + 1).
        """
        return max(self.node_entity.entity_indices.values()) + 1

    @property
    def num_edge_entities(self) -> int:
        """
        Returns number of unique edge entity codes (max index + 1).
        """
        return max(self.edge_entity.entity_indices.values()) + 1

    @property
    def num_edge_specs(self) -> int:
        """
        Returns count of edge_specs entries.
        """
        return len(self.edge_specs)

    @property
    def all_degrees(self) -> list[float]:
        idx_to_name = {idx: name for name, idx in self.node_entity.entity_indices.items()}
        # Make temporary copy of static degrees to avoid modifying the original
        degs = self.node_entity.local_degrees.copy()
        # Update base_edgs with dynamic degs (additive)
        dynamic_degrees = self.node_entity.global_degrees
        for name, deg in dynamic_degrees.items():
            if name in degs:
                degs[name] += deg
            else:
                degs[name] = deg
        return [degs[idx_to_name[i]] for i in range(self.num_node_entities)]

    @property
    def global_degrees(self) -> list[float]:
        degs = self.node_entity.global_degrees
        idx_to_name = {idx: name for name, idx in self.node_entity.entity_indices.items()}
        return [degs[idx_to_name[i]] for i in range(self.num_node_entities) if idx_to_name[i] in degs]

    @property
    def local_degrees(self) -> list[float]:
        degs = self.node_entity.local_degrees
        idx_to_name = {idx: name for name, idx in self.node_entity.entity_indices.items()}
        return [degs[idx_to_name[i]] for i in range(self.num_node_entities) if idx_to_name[i] in degs]

    def get_node_idx(self, node_name: str) -> int:
        """Returns the index of a node entity."""
        try:
            return self.node_entity.entity_indices[node_name]
        except KeyError:
            raise ValueError(f"Node entity '{node_name}' not found.")  # noqa: B904

    def get_edge_idx(self, edge_name: str) -> int:
        """Returns the index of an edge entity."""
        try:
            return self.edge_entity.entity_indices[edge_name]
        except KeyError:
            raise ValueError(f"Edge entity '{edge_name}' not found.")  # noqa: B904

    def get_edge_group_indices(self, group_name: str) -> list[int]:
        """Returns list of edge indices belonging to a named group."""
        try:
            names = self.edge_entity.entity_groups[group_name]
        except KeyError:
            raise ValueError(f"Edge group '{group_name}' not found.")  # noqa: B904
        return sorted({self.edge_entity.entity_indices[name] for name in names})

    def get_node_group_indices(self, group_name: str) -> list[int]:
        """Returns list of node indices belonging to a named group."""
        try:
            names = self.node_entity.entity_groups[group_name]
        except KeyError:
            raise ValueError(f"Node group '{group_name}' not found.")  # noqa: B904
        return sorted({self.node_entity.entity_indices[name] for name in names})

    def get_edge_spec(self, identifier: str | int) -> EdgeSpec:
        """Returns EdgeSpec by name or by index."""
        if isinstance(identifier, int):
            # map index -> name (take first if multiple)
            matches = [name for name, idx in self.edge_entity.entity_indices.items() if idx == identifier]
            if not matches:
                raise ValueError(f"Edge index '{identifier}' not found.")
            name = matches[0]
        else:
            name = identifier
        try:
            return self.edge_specs[name]
        except KeyError:
            raise ValueError(f"Edge spec '{name}' not found.")  # noqa: B904

    def get_edge_specs(self, identifiers: list[str | int], use_scaling: bool = True) -> dict[str, EdgeSpec]:
        """Returns dict of EdgeSpecs for given names or indices."""
        specs: dict[str, EdgeSpec] = {}
        for idf in identifiers:
            spec = self.get_edge_spec(idf)
            # Edit lengths by dimensional scale
            if use_scaling:
                spec = replace(
                    spec,
                    r_max=spec.r_max / self.general.dimensional_scale if spec.r_max is not None else None,
                    start=spec.start / self.general.dimensional_scale if spec.start is not None else None,
                    stop=spec.stop / self.general.dimensional_scale if spec.stop is not None else None,
                    basis_width_scalar=spec.basis_width_scalar / self.general.dimensional_scale
                    if spec.basis_width_scalar is not None
                    else None,
                )
            # determine name key
            if isinstance(idf, int):
                name = next(n for n, i in self.edge_entity.entity_indices.items() if i == idf)
            else:
                name = idf
            specs[name] = spec
        return specs


# --- Instantiate default hyperparameters (comp chemistry only) ---
HPARAMS = HParams(
    node_entity=NodeEntityConfig(
        entity_indices={
            # Ligand
            "ligand_atom": 0,
            "ligand_anchor": 1,
            "ligand_dummy": 2,
            "ligand_virtual": 3,
            # Protein
            "protein_atom": 4,
            "protein_virtual": 5,
        },
        # NOTE: Degrees computed at 5A cutoff pocket and 15A CA cutoff.
        global_degrees={
            # Ligand
            "ligand_atom": 3.3,
            "ligand_anchor": 10,
            "ligand_dummy": 3.5,
            "ligand_virtual": 27,
            # Protein
            "protein_atom": 1.8,
            "protein_virtual": 21,
        },
        # NOTE degrees here are excluding fragment-fragment interactions (optional)!
        # NOTE this is at a 4.0A cutoff
        local_degrees={
            # Updates from static: local_degreees = global + local...
            # Ligand
            "ligand_atom": 6.3,
            "ligand_anchor": 6.1,
            # Protein
            "protein_atom": 1.1,
        },
        entity_groups={
            "ligand": ["ligand_atom", "ligand_anchor", "ligand_dummy", "ligand_virtual"],
            "protein": ["protein_atom", "protein_virtual"],
        },
    ),
    edge_entity=EdgeEntityConfig(
        entity_indices={
            # Ligand edges
            "ligand_bonds": 0,
            "ligand_v2a": 1,
            "ligand_v2v": 2,
            "ligand_torsional_bond": 3,
            "fragment_triangulation": 4,
            "ligand_anchor_dummy": 5,
            # Protein edges
            "protein_bonds": 6,
            "protein_v2v": 7,
            # Protein-Ligand Complex edges
            "complex_lv2pv": 8,
            # Local dynamic interaction edges
            "inter_complex": 9,
            "inter_fragments": 10,
        },
        entity_groups={
            # Core Edges
            "chemistry": ["ligand_bonds", "protein_bonds"],
            "boundary": ["ligand_torsional_bond", "fragment_triangulation", "ligand_anchor_dummy"],
            "virtual": [
                "ligand_v2a",
                "ligand_v2v",
                "complex_lv2pv",
                "protein_v2v",
            ],
            # Static global edges: fixed-length
            "global_static": [
                "ligand_v2a",
                "protein_v2v",
                "ligand_bonds",
                "protein_bonds",
            ],
            # Dynamic global edges: length may vary but no cutoff
            "global_dynamic": [
                "ligand_torsional_bond",
                "fragment_triangulation",
                "ligand_anchor_dummy",
                "ligand_v2v",
            ],
            "local_dynamic": [
                "inter_complex",
                "inter_fragments",
                "complex_lv2pv",
            ],
            # Local interaction edges -> Always dynamic
            "has_cutoff": ["inter_complex", "inter_fragments", "protein_v2v", "complex_lv2pv"],
        },
    ),
    # Note these values are in Angstroms and will be divided by the dimensional_scale when accessing!
    # NOTE: must assert consistency across.
    edge_specs={
        # --- Cutoff edges ---
        # Local interaction smearing with cutoffs
        "inter_complex": EdgeSpec(r_max=4.0, scaling=1 / 8),
        "inter_fragments": EdgeSpec(r_max=4.0, scaling=1 / 8),
        "protein_v2v": EdgeSpec(r_max=18, stop=18, scaling=1 / 8),
        "complex_lv2pv": EdgeSpec(r_max=15, stop=15, scaling=1 / 8),
        # --- Non-Cutoff edges ---
        # Static bond smearing
        "ligand_bonds": EdgeSpec(start=0.0, stop=2.0),
        "protein_bonds": EdgeSpec(start=0.0, stop=2.0),
        # Virtual interactions smearing
        "ligand_v2a": EdgeSpec(start=0.0, stop=4.0),
        "ligand_v2v": EdgeSpec(start=0.0, stop=10.0),
        # Torsional & triangulation (boundary edges) have rel distance as they have relative lengths
        "ligand_anchor_dummy": EdgeSpec(start=0, stop=6.0),
        "ligand_torsional_bond": EdgeSpec(start=0, stop=6.0),
        "fragment_triangulation": EdgeSpec(start=0, stop=6.0),
    },
    general=GeneralConfig(
        dimensional_scale=2.7,  # Å
        epsilon_t=0.01,
    ),
    esm=ESMConfig(
        embedding_dim=1536,
        pos_dim=32,
    ),
)

# TODO add config for protein-ligand parsing as well! Note this changes the DEGREES!
# NOTE optionally we can disentangle this...
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MAX_TORSIONAL_BONDS: int = 20
MAX_WEIGHT: float = 750
