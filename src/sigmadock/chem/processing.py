from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from rdkit import Chem
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from sigmadock.chem import RESIDUE_MAP
from sigmadock.chem.utils import (
    get_coordinates,
    get_fourier_embeddings,
    get_random_rotation_matrix,
)
from sigmadock.oracle import HPARAMS


def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    # Example features: atomic number, degree, formal charge, hybridization, aromaticity
    atomic_number = atom.GetAtomicNum()
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    hybridization = atom.GetHybridization()
    implicit_valence = atom.GetValence(Chem.rdchem.ValenceType.IMPLICIT)
    explicit_valence = atom.GetValence(Chem.rdchem.ValenceType.EXPLICIT)
    is_aromatic = int(atom.GetIsAromatic())

    # Convert hybridization to an integer (for example, sp=1, sp2=2, sp3=3)
    # You might want to define a mapping for consistency.
    hyb_mapping = {
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
        Chem.rdchem.HybridizationType.SP3D2: 5,
    }
    hyb = hyb_mapping.get(hybridization, 0)

    # --- ADD STEREOCHEMISTRY ---
    chiral_tag = atom.GetChiralTag()
    # RDKit mapping: 0=Unspecified, 1=R, 2=S
    # We can use this directly or one-hot encode it.
    # For now, let's add it as a single feature.
    chiral_mapping = {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,  # R
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,  # S
    }
    chiral_feat = chiral_mapping.get(chiral_tag, 0)

    # Convert formal charge to psoitive/negative/neutral
    if formal_charge > 0:
        # Positive Charge
        charge = 2
    elif formal_charge < 0:
        # Negative Charge
        charge = 1
    else:
        # Map neutral charge to 0
        charge = 0

    is_in_ring = int(atom.IsInRing())
    # Get the molecule the atom belongs to, to access ring info
    mol = atom.GetOwningMol()
    num_rings = mol.GetRingInfo().NumAtomRings(atom.GetIdx()) if mol else 0

    # Create a feature vector (you can also use one-hot encodings)
    features = np.array(
        [
            atomic_number,
            degree,
            charge,
            hyb,
            implicit_valence,
            explicit_valence,
            is_aromatic,
            is_in_ring,
            num_rings,
            chiral_feat,
        ],
        dtype=np.float32,
    )
    return features


def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    # Example features: bond type, conjugation, ring membership
    bond_type = bond.GetBondType()
    # Map bond types to integers: SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=4
    bond_type_mapping = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4,
    }
    bt = bond_type_mapping.get(bond_type, 0)
    conjugated = int(bond.GetIsConjugated())
    in_ring = int(bond.IsInRing())

    # --- ADD STEREOCHEMISTRY ---
    stereo_tag = bond.GetStereo()
    # RDKit mapping: 0=None, 1=Any, 2=Z, 3=E
    stereo_mapping = {
        Chem.rdchem.BondStereo.STEREONONE: 0,
        Chem.rdchem.BondStereo.STEREOANY: 1,
        Chem.rdchem.BondStereo.STEREOZ: 2,
        Chem.rdchem.BondStereo.STEREOE: 3,
    }
    stereo_feat = stereo_mapping.get(stereo_tag, 0)
    # Adds bond lengths in Angstroms if conf is provided
    features = np.array([bt, conjugated, in_ring, stereo_feat], dtype=np.float32)
    return features


def mol_to_chemical_graph(mol: Chem.Mol) -> dict[str, torch.Tensor]:
    """
    Extracts a molecular graph from an RDKit molecule, assuming that the molecule
    only contains atoms of interest.

    Returns a dictionary with:
      - "atom_features": NumPy array of shape (N, d) with atom features.
      - "bond_features": NumPy array of shape (E, d_bond) with bond features.
      - "edge_index": NumPy array of shape (2, E) containing edge indices.
      - "coords": NumPy array of shape (N, 3) containing 3D coordinates.
    """
    # Pre-compute ring information once for the whole molecule
    Chem.GetSSSR(mol)

    # Assume all atoms in mol are of interest
    atoms = list(mol.GetAtoms())
    # Map original indices to new indices (in case they are not sequential)
    atom_index_map = {atom.GetIdx(): i for i, atom in enumerate(atoms)}

    # Extract atom features
    atom_features = np.array([get_atom_features(atom) for atom in atoms], dtype=np.float32)

    # Get coordinates from the molecule's conformer
    conf = mol.GetConformer()
    coords = np.array(
        [
            [
                conf.GetAtomPosition(atom.GetIdx()).x,
                conf.GetAtomPosition(atom.GetIdx()).y,
                conf.GetAtomPosition(atom.GetIdx()).z,
            ]
            for atom in atoms
        ],
        dtype=np.float32,
    )

    # Extract edge indices and bond features
    edge_index = []
    bond_features = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        # Use the mapping to obtain the new indices
        if begin in atom_index_map and end in atom_index_map:
            i = atom_index_map[begin]
            j = atom_index_map[end]
            # Add edges in both directions for an undirected graph
            edge_index.append([i, j])
            bond_features.append(get_bond_features(bond))
            edge_index.append([j, i])
            bond_features.append(get_bond_features(bond))

    edge_index = np.array(edge_index).T  # Shape becomes (2, E)
    bond_features = np.array(bond_features)

    return {
        "atom_features": torch.from_numpy(atom_features),
        "bond_features": torch.from_numpy(bond_features),
        "edge_index": torch.from_numpy(edge_index),
        "coords": torch.from_numpy(coords),
    }


def get_protein_residue_info(protein: Chem.Mol) -> dict[str, torch.Tensor | list]:
    """
    Extracts residue information from a protein molecule.
    Returns a dictionary with:
      - "residue_features": NumPy array of shape (N, d) with residue features.
      - "residue_indices": NumPy array of shape (N,) with residue indices.
    """
    ca_atom_ids = []
    residues = []
    pdb_types = []
    for atom in protein.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        pdb_types.append(pdb_info.GetName().strip() if pdb_info is not None else "UNK")
        if pdb_info is not None:  # noqa: SIM102
            # The atom name may include extra whitespace, so strip it.
            if pdb_info.GetName().strip() == "CA":
                ca_atom_ids.append(atom.GetIdx())
                # Get the chain identifier and residue number.
                # Chain id might be None if not set, so default to an empty string.
                chain_id = pdb_info.GetChainId().strip() if pdb_info.GetChainId() else ""
                residue_number = pdb_info.GetResidueNumber()
                residues.append((chain_id, residue_number))

    return {
        "ca_indices": torch.tensor(ca_atom_ids),
        "residues": residues,
        "pdb_types": pdb_types,
    }


# ----------------------------------------------------------------
# <------------------------ LOCAL GRAPH <-----------------------
# ----------------------------------------------------------------


def get_lig_idxs(
    node_entity: torch.Tensor,
    just_core: bool = False,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Get a idxs for ligand atoms (core atoms, anchors, dummies).
    Args:
        node_entity (torch.Tensor): Tensor of shape (N,) with node entity flags.
        just_core (bool): If True, return only ligand atoms (not dummies). If False, return all ligand atom entities.
    Returns:
        torch.Tensor: Indices of ligand atoms in the node_entity tensor.
    """
    if mask is None:
        mask = torch.ones_like(node_entity, dtype=torch.bool)
    if just_core:
        return (
            (
                (node_entity == HPARAMS.get_node_idx("ligand_atom"))
                | (node_entity == HPARAMS.get_node_idx("ligand_anchor"))
            )
            & mask
        ).nonzero(as_tuple=True)[0]
    return (
        (
            (node_entity == HPARAMS.get_node_idx("ligand_atom"))
            | (node_entity == HPARAMS.get_node_idx("ligand_anchor"))
            | (node_entity == HPARAMS.get_node_idx("ligand_dummy"))
        )
        & mask
    ).nonzero(as_tuple=True)[0]


def get_protein_idxs(
    node_entity: torch.Tensor,
    virtual: bool = False,
) -> torch.Tensor:
    """
    Get a idxs for protein atoms (core atoms, virtual).
    """
    if virtual:
        return (
            (node_entity == HPARAMS.get_node_idx("protein_atom"))
            | (node_entity == HPARAMS.get_node_idx("protein_virtual"))
        ).nonzero(as_tuple=True)[0]
    return (node_entity == HPARAMS.get_node_idx("protein_atom")).nonzero(as_tuple=True)[0]


# NOTE this works for a single datapoint -> batch_size=1
def get_protein_ligand_edges(
    pos: torch.Tensor,
    node_entity: torch.Tensor,
    cutoff: float,
    lig_just_atoms: bool = True,
    edge_dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Efficiently find all protein-to-ligand atom pairs within cutoff and build bidirectional edges.
    Falls back to torch.cdist if SciPy is unavailable.
    Returns (edge_index, edge_attr, edge_entity).
    """
    lig_idx = get_lig_idxs(node_entity, just_core=lig_just_atoms)
    prot_idx = get_protein_idxs(node_entity, virtual=False)
    if prot_idx.numel() == 0 or lig_idx.numel() == 0:
        return (
            torch.empty((2, 0), dtype=torch.long, device=pos.device),
            torch.empty((0, 3), dtype=pos.dtype, device=pos.device),
            torch.empty((0,), dtype=torch.long, device=pos.device),
        )

    prot_pos = pos[prot_idx]
    lig_pos = pos[lig_idx]

    tree = cKDTree(lig_pos.cpu().numpy())
    pairs = tree.query_ball_point(prot_pos.cpu().numpy(), cutoff)
    src_list, dst_list = [], []
    for i, js in enumerate(pairs):
        for j in js:
            src_list.append(prot_idx[i].item())
            dst_list.append(lig_idx[j].item())
    # # Fallback: full distance matrix
    # dists = torch.cdist(prot_pos, lig_pos)
    # pi, li = torch.where(dists < cutoff)
    # src_list = prot_idx[pi].tolist()
    # dst_list = lig_idx[li].tolist()

    # build bidirectional
    src = torch.tensor(src_list + dst_list, dtype=torch.long, device=pos.device)
    dst = torch.tensor(dst_list + src_list, dtype=torch.long, device=pos.device)
    edge_index = torch.stack([src, dst], dim=0)

    E = edge_index.size(1)
    edge_attr = torch.zeros((E, edge_dim), dtype=pos.dtype, device=pos.device)
    edge_entity = torch.full((E,), HPARAMS.get_edge_idx("inter_complex"), dtype=torch.long, device=pos.device)
    return edge_index, edge_attr, edge_entity


# NOTE this works for a single datapoint -> batch_size=1
def get_inter_fragment_edges(
    pos: torch.Tensor,
    frag_idx_map: torch.Tensor,
    node_entity: torch.Tensor,
    cutoff: float,
    edge_dim: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Connect ligand atoms from different fragments if within cutoff.
    Uses SciPy cKDTree on ligand positions.
    Returns (edge_index, edge_attr, edge_entity).
    """
    # Get all atoms pertaining to the ligand (atoms, anchors, dummies)
    # NOTE edges to&from overconstrained dummies will be removed later!!!
    lig_idx = get_lig_idxs(node_entity)
    if lig_idx.numel() < 2:
        return (
            torch.empty((2, 0), device=pos.device, dtype=torch.long),
            torch.empty((0, edge_dim), device=pos.device, dtype=pos.dtype),
            torch.empty((0,), device=pos.device, dtype=torch.long),
        )

    lig_pos = pos[lig_idx]
    frag_ids = frag_idx_map[lig_idx]

    tree = cKDTree(lig_pos.cpu().numpy())
    pairs = tree.query_ball_tree(tree, cutoff)
    src_list, dst_list = [], []
    for i, js in enumerate(pairs):
        for j in js:
            if i != j and frag_ids[i] != frag_ids[j]:
                src_list.append(lig_idx[i].item())
                dst_list.append(lig_idx[j].item())
    # dists = torch.cdist(lig_pos, lig_pos)
    # i, j = torch.where((dists < cutoff) & (frag_ids.unsqueeze(1) != frag_ids.unsqueeze(0)))
    # src_list = lig_idx[i].tolist()
    # dst_list = lig_idx[j].tolist()

    edge_index = torch.stack(
        [
            torch.tensor(src_list, device=pos.device),
            torch.tensor(dst_list, device=pos.device),
        ],
        dim=0,
    )
    E = edge_index.size(1)
    edge_attr = torch.zeros((E, edge_dim), device=pos.device, dtype=pos.dtype)
    edge_entity = torch.full((E,), HPARAMS.get_edge_idx("inter_fragments"), device=pos.device, dtype=torch.long)
    return edge_index, edge_attr, edge_entity


# NOTE this works for batch and single datapont -> Passing batch.batch
def get_local_interactions_batched(
    *,
    pos: torch.Tensor,
    node_entity: torch.Tensor,
    frag_idx_map: torch.Tensor,
    batch: torch.Tensor,
    cutoff_complex_interactions: float = 4.0,
    cutoff_complex_virtual: float = -1.0,
    cutoff_fragments: float = -1.0,
    lig_just_atoms: bool = True,
    edge_dim: int = 4,  # Default edge feature dimension
) -> dict[str, torch.Tensor]:
    """
    Build local interaction edges on `batch.new_pos`:
      - protein ↔ ligand within cutoff_complex
      - optionally ligand ↔ ligand of different fragment within cutoff_fragments

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) with atom coordinates.
        batch (torch.Tensor, optional): Tensor of shape (N,) mapping atoms to their batch indices.
        node_entity (torch.Tensor): Tensor of shape (N,) with node entity flags.
        frag_idx_map (torch.Tensor): Tensor of shape (N,) mapping atoms to their fragment indices.
        cutoff_complex (float): Cutoff distance for protein-ligand interactions.
        cutoff_fragments (float): Cutoff distance for inter-fragment interactions. If < 0, no inter-fragment edges added
        lig_just_atoms (bool): If True, only consider ligand atoms for complex. If False, consider all ligand entities.

    Returns a dict with keys:
        'edge_index', 'edge_attr', 'edge_entity'
    """
    edge_index_list = []
    edge_attr_list = []
    edge_ent_list = []

    for batch_idx in torch.unique(batch):
        # extract local indices and tensors
        idx_map = torch.nonzero((batch == batch_idx), as_tuple=False).view(-1)
        local_pos = pos[idx_map]
        local_entity = node_entity[idx_map]
        local_frag_idx = frag_idx_map[idx_map]

        # compute interactions on this subgraph
        sub = _get_local_interactions(
            pos=local_pos,
            node_entity=local_entity,
            frag_idx_map=local_frag_idx,
            cutoff_complex_interactions=cutoff_complex_interactions,
            cutoff_complex_virtual=cutoff_complex_virtual,
            cutoff_fragments=cutoff_fragments,
            lig_just_atoms=lig_just_atoms,
            edge_dim=edge_dim,  # Default edge feature dimension
        )

        # sub['edge_index'] is in [0, num_local)
        ei = sub["edge_index"]
        # map local indices back to global:
        global_src = idx_map[ei[0]]
        global_dst = idx_map[ei[1]]
        edge_index_list.append(torch.stack([global_src, global_dst], dim=0))
        edge_attr_list.append(sub["edge_attr"])
        edge_ent_list.append(sub["edge_entity"])

    # concatenate all samples
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty((2, 0), dtype=torch.long)
    edge_ent = torch.cat(edge_ent_list, dim=0) if edge_ent_list else torch.empty((0,), dtype=torch.long)
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.empty((0, edge_dim), dtype=pos.dtype)

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_entity": edge_ent,
    }


def _get_local_interactions(
    pos: torch.Tensor,
    node_entity: torch.Tensor,
    frag_idx_map: torch.Tensor,
    cutoff_complex_interactions: float = -1.0,
    cutoff_complex_virtual: float = -1.0,
    cutoff_fragments: float = -1.0,
    lig_just_atoms: bool = True,
    edge_dim: int = 4,  # Default edge feature dimension
) -> dict[str, torch.Tensor]:
    """
    Build local interaction edges on batch.new_pos:
      - protein ↔ ligand within cutoff_complex
      - optionally ligand ↔ ligand of different fragment within cutoff_fragments

    Args:
        pos (torch.Tensor): Tensor of shape (N, 3) with atom coordinates.
        node_entity (torch.Tensor): Tensor of shape (N,) with node entity flags.
        frag_idx_map (torch.Tensor): Tensor of shape (N,) mapping atoms to their fragment indices.
        cutoff_complex (float): Cutoff distance for protein-ligand interactions.
        cutoff_fragments (float): Cutoff distance for inter-fragment interactions. If < 0, no inter-fragment edges added
        lig_just_atoms (bool): If True, only consider ligand atoms for complex. If False, consider all ligand entities.

    Returns a dict with keys:
        'edge_index', 'edge_attr', 'edge_entity'
    """
    device, dtype = pos.device, pos.dtype
    ei1 = ea1 = ee1 = ei2 = ea2 = ee2 = ei3 = ea3 = ee3 = None

    # --- Add protein-ligand interaction edges ---
    if cutoff_complex_interactions > 0:
        ei1, ea1, ee1 = get_protein_ligand_edges(
            pos,
            node_entity,
            cutoff_complex_interactions,
            lig_just_atoms=lig_just_atoms,
            edge_dim=edge_dim,
        )

    # --- Add inter-fragment edges ---
    if cutoff_fragments > 0:
        ei2, ea2, ee2 = get_inter_fragment_edges(pos, frag_idx_map, node_entity, cutoff_fragments, edge_dim=edge_dim)

    # --- Add protein-virtual interaction edges ---
    if cutoff_complex_virtual > 0:
        pvirt = torch.where(node_entity == HPARAMS.get_node_idx("protein_virtual"))[0]
        lvirt = torch.where(node_entity == HPARAMS.get_node_idx("ligand_virtual"))[0]

        if pvirt.numel() > 0 and lvirt.numel() > 0:
            cross_pairs = torch.cartesian_prod(pvirt, lvirt)
            complex_virt_dists = (pos[cross_pairs[:, 0]] - pos[cross_pairs[:, 1]]).norm(dim=-1)
            dmask = complex_virt_dists < cutoff_complex_virtual
            cross_pairs = cross_pairs[dmask]

            if cross_pairs.numel() > 0:
                ei3 = torch.cat([cross_pairs, torch.flip(cross_pairs, dims=[1])], dim=0).t()
                ee3 = torch.full(
                    (ei3.size(1),),
                    HPARAMS.get_edge_idx("complex_lv2pv"),
                    dtype=torch.long,
                    device=device,
                )
                ea3 = torch.zeros((ei3.size(1), edge_dim), dtype=dtype, device=device)

    # --- Concatenate all existing tensors ---
    edge_tensors = [t for t in [ei1, ei2, ei3] if t is not None and t.shape[1] > 0]
    attr_tensors = [t for t in [ea1, ea2, ea3] if t is not None and t.shape[0] > 0]
    entity_tensors = [t for t in [ee1, ee2, ee3] if t is not None and t.shape[0] > 0]

    edge_index = (
        torch.cat(edge_tensors, dim=1) if edge_tensors else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    edge_attr = (
        torch.cat(attr_tensors, dim=0) if attr_tensors else torch.empty((0, edge_dim), dtype=dtype, device=device)
    )
    edge_entity = (
        torch.cat(entity_tensors, dim=0) if entity_tensors else torch.empty((0,), dtype=torch.long, device=device)
    )

    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_entity": edge_entity,
    }


# ----------------------------------------------------------------
# <------------------------ GLOBAL--GRAPH <------------------------
# ----------------------------------------------------------------


def get_global_ligand_graph(  # noqa
    frag_mol: Chem.Mol,
    virtual_nodes: dict[int, list[dict[str, Any]]],
    frag_atom_idx: torch.Tensor,
    frag_counter: torch.Tensor,
    anchors: list[int],
    dummies: list[int],
    triangulation_indexes: list[dict[int, list[dict]]] | None,
    mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Build a hierarchical ligand graph by adding virtual nodes and connections.

    Args:
        frag_mol (Chem.Mol): Fragmented RDKit molecule with 3D conformer.
        virtual_nodes (dict[int, list[dict]]): Mapping from fragment index to list of virtual node dicts.
            Each dict has keys: 'coords' (np.ndarray of shape (3,)),
            'connected_atoms' (list[int]), 'type' (str).
        frag_atom_idx (torch.Tensor, optional): Tensor of shape (N,) mapping each original atom to its fragment index.
        frag_counter (int, optional): Counter for the current fragment index.
        anchors (list[int]): List of indices for anchor atoms.
        dummies (list[int]): List of indices for dummy atoms.
        triangulation_indexes (list[dict[int, list[dict]]], optional): List of dictionaries for triangulation.
        mask (torch.Tensor, optional): Boolean tensor of shape (N,) to mask out dummy atoms.
        **kwargs: Ignored (for compatibility).

    Returns:
        dict[str, torch.Tensor]: A graph dict with:
            - x: (N+V, d_atom) node features
            - pos: (N+V, 3) coordinates
            - edge_index: (2, E_total) edge indices
            - edge_attr: (E_total, d_bond) edge features
            - node_entity: (N+V, 3) one-hot flags [is_protein, is_ligand, is_virtual]
            - edge_entity: (E_total) flag for edge entity (int)
            - mask: (N+V,) node mask
    """
    # Base chemical graph
    chemical_graph = mol_to_chemical_graph(frag_mol)
    x = chemical_graph["atom_features"]  # (N, d_atom)
    pos = chemical_graph["coords"]  # (N, 3)
    edge_index = chemical_graph["edge_index"]  # (2, E_orig)
    edge_attr = chemical_graph["bond_features"]  # (E_orig, d_bond)
    N, d_atom = x.shape
    E_orig, d_bond = edge_attr.shape

    # Default mask if not provided
    if mask is None:
        mask = torch.ones(N, dtype=torch.bool)

    # Original edge entity: chemistry edges
    edge_entity_orig = torch.ones((E_orig), dtype=torch.long) * HPARAMS.get_edge_idx("ligand_bonds")

    # Flatten virtual nodes
    vnode_coords: list[np.ndarray] = []
    vnode_connections: list[list[int]] = []
    vnode_frag_counter: list[int] = []
    for frag_id, vnode_list in virtual_nodes.items():
        for vnode in vnode_list:
            # only add connections to unmasked atoms
            filtered_atoms = [idx for idx in vnode["connected_atoms"] if mask[idx]]
            if len(filtered_atoms) == 0:
                continue
            vnode_coords.append(vnode["coords"])
            vnode_connections.append(filtered_atoms)
            vnode_frag_counter.append(frag_id)
    V = len(vnode_coords)

    # If no virtual nodes, return original graph with node_entity, edge_entity and provided mask/frag_idx
    if V == 0:
        node_entity = torch.zeros((N, d_atom), dtype=torch.bool)
        node_entity[:, 1] = True  # is_ligand
        return {
            "x": x,
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_entity": node_entity,
            "edge_entity": edge_entity_orig,
            "mask": mask,
            "frag_idx": frag_atom_idx,
            "frag_counter": frag_counter,
        }

    # Create tensors for virtual nodes
    vnode_pos = torch.from_numpy(np.stack(vnode_coords, axis=0)).to(pos)
    vnode_feat = torch.zeros((V, d_atom), dtype=x.dtype)

    # Concatenate node features and positions
    x = torch.cat([x, vnode_feat], dim=0)  # (N+V, d_atom)
    pos = torch.cat([pos, vnode_pos], dim=0)  # (N+V, 3)
    # Node entity flags
    node_entity = torch.ones((N + V), dtype=torch.long) * HPARAMS.get_node_idx("ligand_atom")
    node_entity[N:] = HPARAMS.get_node_idx("ligand_virtual")  # virtual nodes
    # Anchors & Dummies
    if len(anchors) > 0:
        node_entity[anchors] = HPARAMS.get_node_idx("ligand_anchor")  # anchors
    if len(dummies) > 0:
        node_entity[dummies] = HPARAMS.get_node_idx("ligand_dummy")  # dummies
    assert set(anchors).isdisjoint(set(dummies)), "Anchors and dummies must be disjoint sets."

    # Extend mask for virtual nodes
    mask = torch.cat([mask, torch.ones(V, dtype=torch.bool)], dim=0)

    # Extend fragment index
    vnode_frag_counter_tensor = torch.tensor(vnode_frag_counter, dtype=frag_counter.dtype)
    frag_counter = torch.cat([frag_counter, vnode_frag_counter_tensor], dim=0)
    vnode_frag_atom_idx = torch.arange(N, N + V, dtype=frag_atom_idx.dtype)
    frag_atom_idx = torch.cat([frag_atom_idx, vnode_frag_atom_idx], dim=0)

    # Prepare new edges
    new_edges: list[list[int]] = []
    new_edge_attr: list[torch.Tensor] = []
    new_edge_entity: list[torch.Tensor] = []

    # Virtual-to-atom edges
    for vi, atom_list in enumerate(vnode_connections):
        v_idx = N + vi
        for a in atom_list:
            new_edges.append([v_idx, a])
            new_edges.append([a, v_idx])
            new_edge_attr.extend([torch.zeros(d_bond, dtype=edge_attr.dtype)] * 2)
            new_edge_entity.extend(
                [
                    torch.tensor(HPARAMS.get_edge_idx("ligand_v2a"), dtype=torch.long),
                    torch.tensor(HPARAMS.get_edge_idx("ligand_v2a"), dtype=torch.long),
                ]
            )

    # Virtual-to-virtual edges (global)
    for i in range(V):
        for j in range(i + 1, V):
            vi = N + i
            vj = N + j
            new_edges.append([vi, vj])
            new_edges.append([vj, vi])
            new_edge_attr.extend([torch.zeros(d_bond, dtype=edge_attr.dtype)] * 2)
            new_edge_entity.extend(
                [
                    torch.tensor(HPARAMS.get_edge_idx("ligand_v2v"), dtype=torch.long),
                    torch.tensor(HPARAMS.get_edge_idx("ligand_v2v"), dtype=torch.long),
                ]
            )

    # Add torsional bond & triangulation edges between anchors of one frag to the connected fragment.
    # TODO do the same from or else it creates an asymmetry!!!
    if triangulation_indexes is None:
        tri_map = {}
    else:
        assert len(triangulation_indexes) == 1, "Currently only support one triangulation index set."
        tri_map = triangulation_indexes[0] 
    for src, mappings in tri_map.items():
        for m in mappings:
            # a) connect reference ↔ each linked atom
            for dst in m["linked_atoms"]:
                for u, v in [(src, dst), (dst, src)]:
                    new_edges.append([u, v])
                    new_edge_attr.append(torch.zeros(d_bond, dtype=edge_attr.dtype))
                    new_edge_entity.append(torch.tensor(HPARAMS.get_edge_idx("fragment_triangulation")))
            # b) connect anchor ↔ anchor_neighbor (equivalent to "bond length")
            # NOTE triangulation replicates torsional edges from the original graph, but we need to add them.
            # NOTE it's fine since this is actually a super important message.
            neigh = m["neighbor"]
            for u, v in [(src, neigh), (neigh, src)]:
                new_edges.append([u, v])
                new_edge_attr.append(torch.zeros(d_bond, dtype=edge_attr.dtype))
                new_edge_entity.append(torch.tensor(HPARAMS.get_edge_idx("ligand_torsional_bond")))
            # c) connect anchor -> dummy at that coordinate (0-distance)
            neigh_dummies = dummies[anchors == neigh]
            for neigh_dummy in neigh_dummies:
                if pos[neigh_dummy].allclose(pos[src]):
                    for u, v in [(src, neigh_dummy), (neigh_dummy, src)]:
                        new_edges.append([u, v])
                        new_edge_attr.append(torch.zeros(d_bond, dtype=edge_attr.dtype))
                        new_edge_entity.append(torch.tensor(HPARAMS.get_edge_idx("ligand_anchor_dummy")))

    # Combine with original edges
    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()  # (2, E_new)
    new_edge_attr = torch.stack(new_edge_attr, dim=0)  # (E_new, d_bond)
    new_edge_entity = torch.stack(new_edge_entity, dim=0)  # (E_new)

    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, new_edge_attr], dim=0)
    edge_entity = torch.cat([edge_entity_orig, new_edge_entity], dim=0)

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_entity": node_entity,
        "edge_entity": edge_entity,
        "mask": mask,
        "frag_counter": frag_counter,
        "frag_atom_idx": frag_atom_idx,
        "frag_idx_map": frag_counter[torch.argsort(frag_atom_idx)],
    }


def get_global_protein_graph(  # noqa
    protein: Chem.Mol,
    pdb: str | Path,
    distance_cutoff: float | None = None,
    esm_embeddings: Optional[dict[str, torch.Tensor]] = None,
    esm_embeddings_idx: Optional[dict[str, dict[tuple[str, int, str], int]]] = None,
    esm_embeddings_clip_range: Optional[tuple[float, float]] = None,
    esm_embeddings_scaling_factor: float = 1.0,
    verbose: bool = False,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Extracts the global protein graph from an RDKit molecule, assuming that
    the protein pocket only contains atoms of interest. This keeps the original
    molecular bonds and adds edges between alpha-carbon atoms.

    Construct a global protein graph from an RDKit molecule.

    This includes:
      - Atom features and ESM3 embeddings
      - Positional embeddings based on C-alpha distances
      - Chemical bond edges
      - Virtual C-alpha (CA) to CA global edges
      - Entity flags for nodes/edges

    Args:
        protein (Chem.Mol): RDKit protein molecule with 3D conformer.
        pdb (str | Path): Path or identifier for ESM3 embeddings and residue info.
        distance_cutoff (float | None): Optional distance cutoff for virtual edges.
        esm_embeddings (Optional): If not None, use cached ESM3 embeddings. Default is None.

    Returns:
        Dict[str, Tensor]: Graph data including:
            x (N, d_atom): atom feature matrix
            esm_embeddings (N, d_emb): per-atom ESM3 embeddings
            positional_embeddings (N, d_pe): Fourier-encoded CA distances
            pos (N, 3): 3D coordinates
            edge_index (2, E): edge list
            edge_attr (E, d_bond): edge feature matrix
            node_entity (N): node entity flags
            edge_entity (E): edge entity flags
            mask (N,): boolean mask for completeness with ligand. Default includes all atoms.
    """
    # Assuming all atoms in mol are of interest (already processed!)
    # --- Extract base atom data ---
    atoms = list(protein.GetAtoms())
    N = len(atoms)

    # Extract atom features
    x = torch.tensor([get_atom_features(atom) for atom in atoms])  # (N, d_atom)

    # Extract ESM3 embeddings
    if esm_embeddings is None:
        esm_embeddings_per_atom = None
        # print("[INFO] Using default atom features instead of ESM3 embeddings.")
        # esm_embeddings_per_atom = torch.zeros((len(atoms), 1536))
    else:
        from sigmadock.chem.extract_esm_embeddings import (
            get_esm_embedding_for_atom,
        )  # to prevent circular import

        # FIXME: issue when pdb is str
        esm_embeddings_per_atom = torch.stack(
            [
                get_esm_embedding_for_atom(
                    esm_embeddings[pdb.parent.stem],
                    esm_embeddings_idx[pdb.parent.stem],
                    atom,
                )
                for atom in atoms
            ]
        )
        if esm_embeddings_clip_range is not None:
            esm_embeddings_per_atom = torch.clip(
                esm_embeddings_per_atom,
                min=esm_embeddings_clip_range[0],
                max=esm_embeddings_clip_range[1],
            )
        esm_embeddings_per_atom *= esm_embeddings_scaling_factor

    # Positional embeddings via shortest-path to CA
    # Identify CA atoms
    ca_map: dict[tuple[str, int], tuple[int, int]] = {}
    for i, atom in enumerate(atoms):
        info = atom.GetPDBResidueInfo()
        if info and info.GetName().strip() == "CA":
            key = (info.GetChainId(), info.GetResidueNumber())
            ca_map[key] = (atom.GetIdx(), i)
    # Get shortest path embeddings between atoms and their corresponding CA atom
    # Compute path lengths
    residue_depth = []
    for atom in atoms:
        info = atom.GetPDBResidueInfo()
        idx = atom.GetIdx()
        key = (info.GetChainId(), info.GetResidueNumber())
        ca_idx = ca_map[key][0]
        if ca_idx == idx:
            residue_depth.append(0)
        else:
            length = len(Chem.rdmolops.GetShortestPath(protein, ca_idx, idx))
            residue_depth.append(length)
    residue_depth = torch.tensor(residue_depth, dtype=torch.long)
    positional_embeddings = get_fourier_embeddings(residue_depth, sigma=1 / 1024, num_features=16)

    # Get coordinates from the molecule's conformer
    pos = torch.from_numpy(get_coordinates(protein))  # (N, 3)

    # Map original indices to new indices (in case they are not sequential)
    atom_index_map = {atom.GetIdx(): i for i, atom in enumerate(atoms)}

    # ------------ Chemical Edges ------------
    edge_index, bond_features = [], []
    for bond in protein.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Use the mapping to obtain the new indices
        if begin in atom_index_map and end in atom_index_map:
            i, j = atom_index_map[begin], atom_index_map[end]
            # Add edges in both directions for an undirected graph
            edge_index += [[i, j], [j, i]]
            bond_features += 2 * [get_bond_features(bond)]
    edge_index_chem = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr_chem = torch.from_numpy(np.stack(bond_features, axis=0))
    # Chemistry edge_entity
    E_chem, d_bond = edge_attr_chem.shape
    edge_entity_chem = torch.ones((E_chem), dtype=torch.long) * HPARAMS.get_edge_idx("protein_bonds")

    # --- Virtual CA-CA edges ---
    ca_nodes = [new_i for (_, new_i) in ca_map.values()]
    virtual_pairs = list(combinations(ca_nodes, 2))
    edge_idx_v = []
    for i, j in virtual_pairs:
        # Check distance cutoff
        if distance_cutoff is not None and distance_cutoff > 0:
            # Calculate distance between CA atoms
            dist = torch.norm(pos[i] - pos[j])
            if dist > distance_cutoff:
                # Skip if distance exceeds cutoff
                continue
        edge_idx_v += [[i, j], [j, i]]
    if edge_idx_v:
        edge_index_v = torch.tensor(edge_idx_v, dtype=torch.long).t()
        edge_attr_v = torch.zeros((edge_index_v.shape[1], d_bond), dtype=edge_attr_chem.dtype)
        edge_entity_v = torch.ones((edge_index_v.shape[1]), dtype=torch.long) * HPARAMS.get_edge_idx("protein_v2v")
    else:
        edge_index_v = torch.empty((2, 0), dtype=torch.long)
        edge_attr_v = torch.empty((0, d_bond), dtype=edge_attr_chem.dtype)
        edge_entity_v = torch.empty((0, 0), dtype=torch.bool)

    # combine edges
    edge_index = torch.cat([edge_index_chem, edge_index_v], dim=1)
    edge_attr = torch.cat([edge_attr_chem, edge_attr_v], dim=0)
    edge_entity = torch.cat([edge_entity_chem, edge_entity_v], dim=0)

    # --- Node entity, mask, frag_idx ---
    node_entity = torch.ones((N), dtype=torch.bool) * HPARAMS.get_node_idx("protein_atom")
    for _, i in ca_map.values():
        node_entity[i] = HPARAMS.get_node_idx("protein_virtual")  # virtual nodes

    # Residue types
    residue_types = []
    for atom in protein.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info:
            res_name = pdb_info.GetResidueName().strip()
            residue_idx = RESIDUE_MAP.get(res_name, RESIDUE_MAP["UNK"])
            residue_types.append(residue_idx)
        else:
            residue_types.append(RESIDUE_MAP["UNK"])
    residue_types_tensor = torch.tensor(residue_types, dtype=torch.long)

    return {
        "x": x,
        "pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_entity": node_entity,
        "edge_entity": edge_entity,
        "esm_embeddings": esm_embeddings_per_atom,
        "positional_embeddings": positional_embeddings,
        "mask": torch.ones(N, dtype=torch.bool),
        "residue_types": residue_types_tensor,
    }


def get_global_interaction_graph(
    protein_graph: Data,
    ligand_graph: Data,
    prot_coordinate_noise: float | None = None,
    lig_coordinate_noise: float | None = None,
    random_rotation: bool = False,
    pocket_com: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Merge protein and ligand graphs into a single interaction graph.

    Connect all virtual nodes across protein and ligand.

    Args:
        protein_graph (Dict[str, Tensor]): Protein graph dict with keys:
            'x', 'pos', 'edge_index', 'edge_attr', 'node_entity', 'edge_entity', 'mask', ...
        ligand_graph (Dict[str, Tensor]): Ligand graph dict with same keys + "frag_idx".
        prot_coordinate_noise (float | None): Optional noise to add to protein coordinates.
        lig_coordinate_noise (float | None): Optional noise to add to ligand coordinates.
        random_rotation (bool): If True, apply random rotation to the system.
        pocket_com (torch.Tensor | None): The coordinate vector of the center of mass to rotate the system around.
            Required if random_rotation is True.

    Returns:
        Dict[str, Tensor]: Combined graph dict with:
            # Concatenated
            - x: concatenated node features
            - pos: concatenated positions
            - edge_index: merged edges (chem, virtual within, and cross virtual-to-virtual)
            - edge_attr: concatenated edge attributes
            - node_entity: concatenated node entity flags
            - edge_entity: concatenated edge entity flags
            - mask: concatenated masks
            - frag_idx: concatenated fragment indices (ligand-only, protein is -1)
            # Protein Only
            - protein_embeddings: [Dict[str, Tensor]] with 'esm_embeddings' and 'positional_embeddings'
    """
    # ----------------  Complex ----------------
    # NOTE only build global things here, local vicinities need to be built at each timestep in diffusion

    # ---------------- GRAPH ----------------

    # Unpack protein
    px = protein_graph["x"]
    ppos = protein_graph["pos"]
    p_emb = protein_graph.get("esm_embeddings")
    p_posenc = protein_graph.get("positional_embeddings")
    pei = protein_graph["edge_index"]
    pea = protein_graph["edge_attr"]
    pne = protein_graph["node_entity"]
    pee = protein_graph["edge_entity"]
    p_res_type = protein_graph["residue_types"]
    pmask = protein_graph["mask"]

    # Unpack ligand
    lx = ligand_graph["x"]
    lpos = ligand_graph["pos"]
    lei = ligand_graph["edge_index"]
    lea = ligand_graph["edge_attr"]
    lne = ligand_graph["node_entity"]
    lee = ligand_graph["edge_entity"]
    lmask = ligand_graph["mask"]
    lfrag_counter = ligand_graph["frag_counter"]
    lfrag_idx = ligand_graph["frag_atom_idx"]
    lfrag_map = ligand_graph["frag_idx_map"]

    # Residue Types
    lig_res_type_pad = torch.full((lx.size(0),), RESIDUE_MAP["UNK"], dtype=p_res_type.dtype)
    residue_types = torch.cat([p_res_type, lig_res_type_pad], dim=0)

    # Concatenate node features and metadata
    x = torch.cat([px, lx], dim=0)

    # Add noise to coordinates if specified (protein and ligand)
    if lig_coordinate_noise is not None:
        lnoise = torch.randn_like(lpos) * lig_coordinate_noise
        lpos += lnoise
    if prot_coordinate_noise is not None:
        pnoise = torch.randn_like(ppos) * prot_coordinate_noise
        ppos += pnoise

    pos = torch.cat([ppos, lpos], dim=0)
    if random_rotation:
        assert pocket_com is not None, "If random_rotation is True, pocket_com must be provided."
        # Randomly rotate the ligand coordinates around the origin
        rotation_matrix = get_random_rotation_matrix(device=pos.device, dtype=pos.dtype)
        pos = torch.matmul(pos - pocket_com, rotation_matrix) + pocket_com

    node_entity = torch.cat([pne, lne], dim=0)
    mask = torch.cat([pmask, lmask], dim=0)

    # frag_counter: protein = -1, ligand retains
    prot_frag = torch.full((px.size(0),), -1, dtype=lfrag_counter.dtype)
    frag_counter = torch.cat([prot_frag, lfrag_counter], dim=0)
    # frag_idx: protein = -1, ligand retains
    frag_idx = torch.full((px.size(0),), -1, dtype=lfrag_idx.dtype)
    frag_idx = torch.cat([frag_idx, lfrag_idx], dim=0)
    # frag_map: protein = -1, ligand retains
    prot_frag_map = torch.full((px.size(0),), -1, dtype=lfrag_map.dtype)
    frag_map = torch.cat([prot_frag_map, lfrag_map], dim=0)

    # Merge edges
    Np = px.size(0)
    lei_shifted = lei + Np
    edge_index = torch.cat([pei, lei_shifted], dim=1)
    edge_attr = torch.cat([pea, lea], dim=0)
    edge_entity = torch.cat([pee, lee], dim=0)

    result = {
        "x": x,
        "ref_pos": pos,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_entity": node_entity,
        "edge_entity": edge_entity,
        "mask": mask,
        "frag_atom_idx": frag_idx,
        "frag_counter": frag_counter,
        "frag_idx_map": frag_map,
        "residue_types": residue_types,
        # NOTE we do not concatenate with dummy ligand embeddings for computational efficiency.
        "protein_embeddings": {
            "esm_embeddings": p_emb,
            "positional_embeddings": p_posenc,
        },
    }
    return result
