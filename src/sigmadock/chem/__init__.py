EDGE_ENTITY: dict[str, int] = {
    # NOTE GLOBAL
    # Ligand
    "ligand_bonds": 0,
    "ligand_a2v": 1,
    "ligand_v2a": 1,
    "ligand_v2v": 2,
    "ligand_torsional_bond": 3,  # Literally the torsional bond separated by distance we know.
    "fragment_triangulation": 4,  # Triangulation from anchors to anchor neighbours separated by distance.
    "ligand_anchor_dummy": 5,  # 0-distance edges between matching anchors and dummy atoms
    # Protein
    "protein_bonds": 6,
    "protein_v2v": 7,
    # Protein-Ligand Complex
    "complex_pv2lv": 8,
    "complex_lv2pv": 8,
    # NOTE LOCAL
    "inter_complex": 9,
    "inter_fragments": 10,
}
NODE_ENTITY: dict[str, int] = {
    # Ligand
    "is_ligand_atom": 0,
    "is_ligand_anchor": 1,
    "is_ligand_dummy": 2,
    "is_ligand_virtual": 3,
    # Protein
    "is_protein_atom": 4,
    "is_protein_virtual": 5,  # NOTE this is basically is CA
}

# HYPERPARAMETERS
DIMENSIONAL_SCALE: float = 2.7  # STD scale for dimensionality reduction on ligand fragment centroids.

# Cutoffs
DEFAULT_CUTOFFS = {
    "fragments": -1.0,  # No cutoff
    "complex": 4.0,  # 4A between protein and ligand atoms
    "complex_skip_dummy_anchor_interactions": True,  # Skip ligand dummy anchors interactions with protein.
}

RESIDUE_MAP = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "UNK": 20,  # For unknown or non-standard residues
}
