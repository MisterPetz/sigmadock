NODE_ENTITY_SETS: dict[str, list[int]] = {
    "total_entities": 6,
    "ligand": [0, 1, 2, 3],
    "virtual_ligand": [3],
    "protein": [4, 5],
    "virtual_protein": [5],
}

# for r_max of cutoff
LOCAL_EDGE_DISTANCE_SETS: dict[int, dict[str, float]] = {
    7: {"r_max": 5.0},  # ca to ca
    9: {"r_max": 5.0},  # ligand to ligand
    10: {"r_max": 5.0},  # ligand to protein
}

GLOBAL_EDGE_DISTANCE_SETS: dict[int, dict[str, float]] = {
    0: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # ligand to ligand
    1: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # ligand to virtual
    2: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # virtual to virtual
    3: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # torsional
    4: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # triangulation
    5: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # anchor to dummy
    6: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # protein to protein
    8: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # protein to virtual
}

# for edge features of boundary edges; we use GaussianSmearing
BOUNDARY_EDGE_FEATURE_SETS: dict[int, dict[str, float]] = {
    3: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # torsional
    4: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # triangulation
    5: {"start": 0.0, "stop": 5.0, "basis_width_scalar": 0.5},  # anchor to dummy
}

EDGE_ENTITY_SETS = {
    "total_entities": 11,
    "chemistry": [0, 6],
    "boundary": BOUNDARY_EDGE_FEATURE_SETS,
    "local": LOCAL_EDGE_DISTANCE_SETS,
    "local_dynamic": [9, 10],  # for MLPs
    "global": GLOBAL_EDGE_DISTANCE_SETS,
}

ESM_EMBEDDING_DIM = 1536
ESM_POS_DIM = 32
