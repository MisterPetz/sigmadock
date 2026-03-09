"""
fragmentation.py

A module for fragmenting molecules using RDKit.
Includes candidate bond detection, dummy insertion, safe fragmentation,
state validity checking, recursive merge (cut removal), trivial fragment merging,
and visualization functions for valid fragmentation states.
"""

import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence as Seq
from pathlib import Path
from typing import Callable, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, TorsionFingerprints
from rdkit.Geometry import Point3D  # noqa
from rdkit.rdBase import BlockLogs
from functools import lru_cache # noqa
from typing import Iterable

# ---------------------------------------------------------------
# 1. Candidate Bond Detection
# ---------------------------------------------------------------


def detect_torsional_bonds(mol: Chem.Mol, ignore_conjugated: bool = False) -> set[int]:
    """
    Detect non-ring torsional bonds in a molecule.

    Parameters:
        mol (Chem.Mol): The RDKit molecule.
        ignore_conjugated (bool): If True, skip conjugated bonds.

    Returns:
        set[int]: A set of bond indices corresponding to rotatable (candidate) bonds.
    """
    Chem.SanitizeMol(mol)
    non_ring_tors, _ = TorsionFingerprints.CalculateTorsionLists(mol)
    torsional_bonds: set[int] = set()
    for torsion_group in non_ring_tors:
        torsion_quads, _ = torsion_group
        for quad in torsion_quads:
            a2, a3 = quad[1], quad[2]
            bond = mol.GetBondBetweenAtoms(a2, a3)
            if bond is None:
                continue
            if bond.IsInRing():
                continue
            if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
                continue
            if bond.GetIsConjugated() and ignore_conjugated:  # noqa: SIM102
                if not mol.GetAtomWithIdx(a2).IsInRing() and not mol.GetAtomWithIdx(a3).IsInRing():
                    # Only skip conjugated bonds that are not in / stemming from rings (weak conjugation)
                    continue
            torsional_bonds.add(bond.GetIdx())
    return torsional_bonds


def identify_all_cuttable_bonds(mol: Chem.Mol, ignore_conjugated: bool = False) -> list[int]:
    """
    Identify all torsional bonds that should be cut to ensure that no
    rotational bonds exist within any fragment.

    Parameters:
        mol (Chem.Mol): The RDKit molecule.
        ignore_conjugated (bool): If True, skip conjugated bonds.

    Returns:
        list[int]: A list of bond indices considered cuttable.
    """
    torsional_bonds = detect_torsional_bonds(mol, ignore_conjugated)
    return list(torsional_bonds)


# ---------------------------------------------------------------
# 2. Fragmentation and Dummy Insertion
# ---------------------------------------------------------------


def fragment_on_bonds(mol: Chem.Mol, cuttable_bonds: list[int]) -> Chem.Mol:
    """
    Fragment the molecule at the specified bonds and insert dummy atoms at the cut points.
    Then, assign pair labels (atom map numbers) to dummy atoms in pairs.

    Parameters:
        mol (Chem.Mol): The RDKit molecule.
        cuttable_bonds (list[int]): The bond indices at which to fragment.

    Returns:
        Chem.Mol: The fragmented molecule.
    """
    fragmented = Chem.FragmentOnBonds(mol, cuttable_bonds, addDummies=True)
    Chem.SanitizeMol(fragmented)

    # Gather dummy atoms (atomic number 0) and assume they come in pairs.
    dummies = sorted(
        [atom for atom in fragmented.GetAtoms() if atom.GetAtomicNum() == 0],
        key=lambda a: a.GetIdx(),
    )
    if len(dummies) != 2 * len(cuttable_bonds):
        raise ValueError("Unexpected number of dummy atoms found!")

    # Label dummy pairs: assign the same atom map number to each pair.
    label = 1
    for i in range(0, len(dummies), 2):
        dummies[i].SetAtomMapNum(label)
        if i + 1 < len(dummies):
            dummies[i + 1].SetAtomMapNum(label)
        label += 1

    return fragmented


def count_adjacent_non_ring_torsional(
    mol: Chem.Mol,
    anchor_index: int,
    exclude_bond_idxs: list[int],
    ignore_conjugated: bool = False,
) -> int:
    """
    Count adjacent bonds for `anchor_index` that:
      - are torsional (according to detect_torsional_bonds)
      - are not in `exclude_bond_idxs`
      - on the neighbor side of the bond lead to at least one NON-DUMMY, NON-H atom
        via a non-ring bond (also not excluded). This approximates "torsional
        connectivity inside the fragment after removing other dummies".

    Args:
        mol (Chem.Mol): Molecule (pre-fragmentation).
        anchor_index (int): Index of the anchor atom.
        exclude_bond_idxs (list[int]): Bond indices to ignore (the cut bonds).
        ignore_conjugated (bool): Passed to detect_torsional_bonds.

    Returns:
        int: number of adjacent non-ring torsional bonds (as defined above).
    """
    count = 0
    atom = mol.GetAtomWithIdx(int(anchor_index))
    exclude_set = set(int(x) for x in exclude_bond_idxs)  # noqa
    torsional_bonds = set(detect_torsional_bonds(mol, ignore_conjugated))

    for bond in atom.GetBonds():
        bidx = int(bond.GetIdx())
        if bidx in exclude_set:
            continue
        if bidx not in torsional_bonds:
            continue

        # Identify the atom on the other side of this bond
        other = bond.GetOtherAtom(atom)  # RDKit convenience
        # Now check whether `other` connects onward to some heavy non-ring neighbor
        found_valid_path = False
        for nb_b in other.GetBonds():
            nb_idx = int(nb_b.GetIdx())
            # skip the same bond or excluded bonds
            if nb_idx == bidx or nb_idx in exclude_set:
                continue
            nb_other = nb_b.GetOtherAtom(other)
            # skip back to anchor
            if nb_other.GetIdx() == anchor_index:
                continue
            # skip hydrogens and dummies (we don't want those to count as a lever)
            z = nb_other.GetAtomicNum()
            if z == 1 or z == 0:
                continue
            # If we reach here, there is a continuing heavy non-ring neighbor
            found_valid_path = True
            break

        if found_valid_path:
            count += 1

    return count


def fragment_on_bonds_with_mapping(
    orig_mol: Chem.Mol, cuttable_bonds: list[int], ignore_conjugated: bool = False
) -> Chem.Mol:
    """
    Fragment the molecule at the specified bonds, inserting dummy atoms at the cut points.
    Also builds a mapping that includes:
      - "torsional_bonds": a list of dictionaries, one per cut bond, storing:
           - "bond": a tuple (heavy1, heavy2) of the original heavy atom indices,
           - "torsionality": an integer label (0, 1, or 2) computed as follows:
                 0 = both anchors have no adjacent non-ring torsional bonds
                 1 = one anchor has extra non-ring torsional bonds, the other not
                 2 = both anchors have extra non-ring torsional bonds.
      - "anchor_to_dummy": a dictionary mapping each heavy anchor atom index to the corresponding dummy atom index.
      The idea is that for a dummy atom inserted at a break, we want its coordinate to regress toward that heavy atom.

    args:
        orig_mol (Chem.Mol): The original molecule.
        cuttable_bonds (list[int]): List of bond indices to cut.
        ignore_conjugated (bool): If True, skip conjugated bonds.

    Returns:
        Chem.Mol: The fragmented molecule with a JSON property "fragmentation_mapping" attached.
    """
    # Initialize the mapping
    mapping = {"torsional_bonds": [], "anchor_to_dummy": defaultdict(list)}

    if not cuttable_bonds:
        orig_mol.SetProp("fragmentation_mapping", json.dumps(mapping))
        return orig_mol

    # Fragment the molecule (adds dummy atoms)
    fragmented = Chem.FragmentOnBonds(orig_mol, cuttable_bonds, addDummies=True)
    Chem.SanitizeMol(fragmented)
    assert Chem.SanitizeMol(fragmented) == Chem.rdmolops.SANITIZE_NONE, (
        f"Sanitization failed after fragmentation for {Chem.MolToSmiles(fragmented)}!"
    )

    # Get all dummy atoms sorted by index
    dummies = sorted(
        [atom for atom in fragmented.GetAtoms() if atom.GetAtomicNum() == 0],
        key=lambda a: a.GetIdx(),
    )
    # Check that we have the expected number of dummy atoms (2 per cut)
    if len(dummies) != 2 * len(cuttable_bonds):
        raise ValueError("Unexpected number of dummy atoms found!")

    # Merge everything into one loop: process each bond we are cutting.
    for i, bond_idx in enumerate(cuttable_bonds):
        bond = orig_mol.GetBondWithIdx(bond_idx)
        src = bond.GetBeginAtomIdx()
        dst = bond.GetEndAtomIdx()

        # For each anchor, count the additional non-ring torsional bonds adjacent to it
        count_src = count_adjacent_non_ring_torsional(
            orig_mol, src, cuttable_bonds, ignore_conjugated=ignore_conjugated
        )
        count_dst = count_adjacent_non_ring_torsional(
            orig_mol, dst, cuttable_bonds, ignore_conjugated=ignore_conjugated
        )

        # Determine the dofs of each anchor.
        # An anchor is "non-rigid" if its count > 0.
        mapping["torsional_bonds"].append(
            {
                "bond": (src, dst),
                "dofs": [(1 if count_src > 0 else 0), (1 if count_dst > 0 else 0)],
            }
        )

        # Retrieve the two dummy atoms corresponding to this bond.
        dummy_src = dummies[2 * i]  # Assigned to heavy1
        dummy_dst = dummies[2 * i + 1]  # Assigned to heavy2

        # Update the mapping from heavy (anchor) atom index to dummy atom index.
        # append to a list, so a heavy atom can appear multiple times
        mapping["anchor_to_dummy"][src].append(dummy_src.GetIdx())
        mapping["anchor_to_dummy"][dst].append(dummy_dst.GetIdx())

    # Optionally, attach the mapping as a serialized JSON property to the fragmented molecule.
    mapping["anchor_to_dummy"] = dict(mapping["anchor_to_dummy"])
    fragmented.SetProp("fragmentation_mapping", json.dumps(mapping))
    return fragmented


def get_torsional_neighbors(torsional_bonds: list[dict]) -> dict:
    """
    Build a mapping of each anchor to the neighboring anchors it is connected to
    through the fragmentation torsional bonds.

    The input frag_map is assumed to have a key "torsional_bonds",
    which is a list of dictionaries with entries:
        - "bond": a tuple (anchor1, anchor2)
        - other keys (e.g. "dofs") may be present.

    Returns:
        A dictionary mapping each anchor atom index to a list of dictionaries,
        each with the keys:
            - "neighbor": the index of the neighboring anchor,
            - "bond_idx": the index (or identifier) of the torsional bond in the original list.

    Example:
        If frag_map["torsional_bonds"] contains:
            [{"bond": (10, 20), "dofs": [1, 0]}, {"bond": (20, 30), "dofs": [0, 1]}]
        then the output will be:
            {10: [{"neighbor": 20, "bond_idx": 0}],
             20: [{"neighbor": 10, "bond_idx": 0}, {"neighbor": 30, "bond_idx": 1}],
             30: [{"neighbor": 20, "bond_idx": 1}]}
    """
    anchor_neighbors = {}
    for bond_idx, bond in enumerate(torsional_bonds):
        a1, a2 = bond
        # For anchor a1, record neighbor a2
        if a1 not in anchor_neighbors:
            anchor_neighbors[a1] = []
        anchor_neighbors[a1].append({"neighbor": a2, "bond_idx": bond_idx})

        # Likewise, for anchor a2, record neighbor a1
        if a2 not in anchor_neighbors:
            anchor_neighbors[a2] = []
        anchor_neighbors[a2].append({"neighbor": a1, "bond_idx": bond_idx})
    return anchor_neighbors


def get_non_torsional_neighbors(mol: Chem.Mol, anchors: list[int]) -> dict[int, list[int]]:
    anchor_neighs = {}
    for _, a_idx in enumerate(anchors):
        if a_idx not in anchor_neighs:
            anchor_neighs[a_idx] = []
        neighs = [a.GetIdx() for a in mol.GetAtomWithIdx(int(a_idx)).GetNeighbors()]
        # Filter out neighbours that are dummies
        for n in neighs:
            anchor_neighs[a_idx].append(n)
    return anchor_neighs


def get_triangle_equality_mapping(mol: Chem.Mol, neighbors: dict) -> dict:
    """
    Build a mapping of each anchor to its corresponding triangle equality atoms.

    For each anchor, this function examines its neighboring anchors (obtained
    from the fragmentation mapping) and retrieves all atoms bonded to each neighbor,
    excluding the original anchor itself. This mapping can be used to enforce a
    boundary condition in which fixed distances from the anchor to these linked atoms
    help preserve the bond angle (the "triangle equality" condition).

    Args:
        mol (Chem.Mol): The RDKit molecule (typically the fragmented molecule).
        neighbours (dict): A dictionary mapping each atom from the torsioanl bond to a list of neighbor
            entries. Each entry is a dictionary with keys:
                - "neighbor": the neighbor anchor atom index.
                - "bond_idx": an identifier for the bond (if applicable).

    Returns:
        dict: A dictionary mapping each anchor atom index to a list of dictionaries.
              Each inner dictionary has keys:
                  - "anchor_neighbor": the index of a neighboring anchor.
                  - "linked_atoms": a list of atom indices that are directly bonded
                                    to this neighbor (excluding the origin anchor).

    Example:
        Given anchor_neighbors like:
            {10: [{"neighbor": 20, "bond_idx": 0},
                  {"neighbor": 15, "bond_idx": 1}],
             20: [{"neighbor": 10, "bond_idx": 0}],
             15: [{"neighbor": 10, "bond_idx": 1}]}
        The function will produce a mapping such as:
            {10: [{"anchor_neighbor": 20, "linked_atoms": [5, 7, 30]},
                  {"anchor_neighbor": 15, "linked_atoms": [2, 3]}],
             20: [{"anchor_neighbor": 10, "linked_atoms": [8, 9]}],
             15: [{"anchor_neighbor": 10, "linked_atoms": [4, 11]}]}
    """
    triangle_mapping = {}
    # Loop over each anchor in the provided neighbors mapping.
    for src, neighbors_info in neighbors.items():
        triangle_mapping[src] = []
        # Process each neighbor entry for this anchor.
        for neighbor_entry in neighbors_info:
            neighbor = neighbor_entry["neighbor"]
            neighbor_atom = mol.GetAtomWithIdx(int(neighbor))
            # Gather all atoms bonded to the neighbor except the origin anchor.
            linked_atoms = [nbr.GetIdx() for nbr in neighbor_atom.GetNeighbors() if nbr.GetIdx() != src]
            triangle_mapping[src].append({"neighbor": neighbor, "linked_atoms": linked_atoms})
    return triangle_mapping


def safe_fragment(orig_mol: Chem.Mol, cut_set: set[int]) -> list[Chem.Mol]:
    """
    Fragment the molecule at the bonds specified by cut_set.
    If cut_set is empty, returns the intact molecule as a single-item list.

    Parameters:
        orig_mol (Chem.Mol): The original molecule.
        cut_set (set[int]): Set of bond indices to cut.

    Returns:
        list[Chem.Mol]: A list of fragment molecules.
    """
    cut_list = sorted(cut_set)
    if not cut_list:
        return [orig_mol]
    frag_mol = fragment_on_bonds_with_mapping(orig_mol, cut_list)
    return Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)


def get_fragment_map(frag_mol: Chem.Mol) -> dict[int, int]:
    """
    Get fragment map as dict from the molecule.
    The mapping is stored in the 'fragmentation_mapping' property of the molecule.
    The mapping contains the torsional bonds and the mapping from heavy atoms to dummy atoms.

    args:
        frag_mol (Chem.Mol): The fragmented molecule.
    Returns:
        dict[str, list[int] | dict[int, int]]: A dictionary mapping heavy atom indices to dummy atom indices.
    """
    frag_map = json.loads(frag_mol.GetProp("fragmentation_mapping"))
    frag_map["anchor_to_dummy"] = {int(k): v for k, v in frag_map["anchor_to_dummy"].items()}
    return frag_map


# ---------------------------------------------------------------
# 3. Clean Fragments: Remove Dummy Atoms
# ---------------------------------------------------------------


def remove_dummy_atoms(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove all dummy atoms (atomic number 0) from a molecule.
    If kekulization fails during sanitization, re-sanitize while suppressing kekulization.

    Parameters:
        mol (Chem.Mol): The molecule to clean.

    Returns:
        Chem.Mol: The cleaned molecule (without dummy atoms).
    """
    em = Chem.EditableMol(mol)
    dummy_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in sorted(dummy_idxs, reverse=True):
        em.RemoveAtom(idx)
    cleaned = em.GetMol()

    # Create a Blocker object to suppress logs
    blocker = BlockLogs()
    try:
        # First attempt at sanitization (warnings are suppressed)
        Chem.SanitizeMol(cleaned)
    except Exception:
        # Second attempt if the first fails (warnings are also suppressed). Not a big deal if this doesn't work because
        # we are just checking if the fragmnets are valid.
        Chem.SanitizeMol(
            cleaned,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    finally:
        # This guarantees logs are re-enabled, even if both attempts fail
        del blocker
    return cleaned


# ---------------------------------------------------------------
# 4. State Validity: Check Fragmentation for Torsional Bonds
# ---------------------------------------------------------------

# NOTE: Deprecated
def _state_is_valid(
    orig_mol: Chem.Mol,
    cut_set: set[int],
    ignore_conjugated: bool = False,
    final_state_check: bool = False,
) -> bool:
    frags = safe_fragment(orig_mol, cut_set)
    for frag in frags:
        clean_frag = remove_dummy_atoms(frag)
        # Fragments with 3 or fewer atoms are too small to have rotatable bonds
        if clean_frag.GetNumAtoms() > 3:  # noqa
            if detect_torsional_bonds(clean_frag, ignore_conjugated):
                return False
        if final_state_check:  # noqa: SIM102
            # Ensure no fragments have more than 1 atom (undefined rototranslations)
            # Ideally more than 2 atoms because some may have overconstrained torsional bonds
            if clean_frag.GetNumAtoms() < 2:
                return False
    return True


# Global cache across calls: key is canonical SMILES of a fragment and ignore_conjugated
_frag_torsion_cache: dict[tuple[str, bool], bool] = {}


def fragment_has_torsions_mol(clean_frag: Chem.Mol, ignore_conjugated: bool) -> bool:
    """
    Avoids MolFromSmiles roundtrip. SMILES used only as cache key.
    """
    smi = Chem.MolToSmiles(clean_frag, isomericSmiles=True, canonical=True)
    key = (smi, ignore_conjugated)
    hit = _frag_torsion_cache.get(key)
    if hit is not None:
        return hit
    has = any(detect_torsional_bonds(clean_frag, ignore_conjugated))
    _frag_torsion_cache[key] = has
    return has


def state_is_valid(
    orig_mol: Chem.Mol,
    cut_set: set[int] | frozenset[int],
    ignore_conjugated: bool = False,
    final_state_check: bool = False,
) -> bool:
    """
    Check if fragmenting the molecule with the given cut_set results in fragments that
    (if sufficiently large) do not contain any candidate torsional bonds.

    Parameters:
        orig_mol (Chem.Mol): The original molecule.
        cut_set (set[int]): Set of bond indices to cut.
        ignore_conjugated (bool): If True, skip conjugated bonds during torsion detection.
        final_state_check (bool): If True, only check the final state after fragmentation.

    Returns:
        bool: True if all sufficiently large fragments have no torsional bonds.
    """
    frags = safe_fragment(orig_mol, set(cut_set))
    for frag in frags:
        clean_frag = remove_dummy_atoms(frag)
        n = clean_frag.GetNumAtoms()

        if final_state_check and n < 2:
            return False

        if n > 3 and fragment_has_torsions_mol(clean_frag, ignore_conjugated):
            return False

    return True


# ------------------------------------------------------------
# Helpers: torsion candidate classification
# ------------------------------------------------------------

def _compute_degT_and_endpoints(
    mol: Chem.Mol,
    torsion_bonds: Iterable[int],
    *,
    exclude_ring_bonds_from_deg: bool = True,
) -> tuple[dict[int, int], dict[int, tuple[int, int]], dict[int, bool]]:
    """
    degT[a] = number of candidate torsion bonds incident to atom a
    endpoints[b] = (u, v)
    in_ring[b] = bond.IsInRing()
    """
    degT: dict[int, int] = defaultdict(int)
    endpoints: dict[int, tuple[int, int]] = {}
    in_ring: dict[int, bool] = {}

    for bidx in torsion_bonds:
        b = mol.GetBondWithIdx(bidx)
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        endpoints[bidx] = (u, v)
        in_ring[bidx] = b.IsInRing()

    for bidx, (u, v) in endpoints.items():
        if exclude_ring_bonds_from_deg and in_ring[bidx]:
            continue
        degT[u] += 1
        degT[v] += 1

    return degT, endpoints, in_ring


def _biaryl_like_bond(mol: Chem.Mol, bidx: int) -> bool:
    """
    Deterministic "two aromatic rings connected" heuristic.

    True if:
      - single bond
      - not in ring
      - both atoms aromatic
    """
    b = mol.GetBondWithIdx(bidx)
    if b.IsInRing():
        return False
    if b.GetBondType() != Chem.BondType.SINGLE:
        return False
    a1 = b.GetBeginAtom()
    a2 = b.GetEndAtom()
    return bool(a1.GetIsAromatic() and a2.GetIsAromatic())


def deterministic_prune_bonds(
    mol: Chem.Mol,
    torsion_bonds: Iterable[int],
    *,
    exclude_ring_bonds: bool = True,
    force_cut_biaryl: bool = False,
    force_cut_ring_bridges: bool = True, # New flag
) -> tuple[set[int], list[int], dict[int, tuple[int, int]]]:
    
    torsion_bonds = sorted(set(torsion_bonds))
    degT, endpoints, in_ring = _compute_degT_and_endpoints(mol, torsion_bonds)

    must_cut: set[int] = set()
    removable: list[int] = []

    for bidx in torsion_bonds:
        # 1. Ring bonds (if excluded)
        if exclude_ring_bonds and in_ring[bidx]:
            must_cut.add(bidx)
            continue

        # 2. Biaryl-specific (Aromatic-Aromatic)
        if force_cut_biaryl and _biaryl_like_bond(mol, bidx):
            must_cut.add(bidx)
            continue

        u_idx, v_idx = endpoints[bidx]
        u_atom = mol.GetAtomWithIdx(u_idx)
        v_atom = mol.GetAtomWithIdx(v_idx)

        # 3. (Connecting 2 ring systems)
        # If both atoms are in rings, this bond is the only way to separate the rings.
        if force_cut_ring_bridges and u_atom.IsInRing() and v_atom.IsInRing():
            must_cut.add(bidx)
            continue

        # 4. Isolated Torsion (Original Logic)
        if degT.get(u_idx, 0) == 1 and degT.get(v_idx, 0) == 1:
            must_cut.add(bidx)
        else:
            removable.append(bidx)

    removable.sort()
    return must_cut, removable, endpoints

# ---------------------------------------------------------------
# 5. Recursive Merge / Cut Removal
# ---------------------------------------------------------------

# NOTE: Deprecated
def _recursive_merge(
    orig_mol: Chem.Mol,
    current_cut_set: set[int],
    candidate_list: list[int],
    start: int,
    ignore_conjugated: bool = False,
) -> set[frozenset[int]]:
    """
    Recursively attempt to remove bonds from current_cut_set (merging fragments)
    such that the resulting fragmentation state is valid.

    Parameters:
        orig_mol (Chem.Mol): The original molecule.
        current_cut_set (set[int]): The current set of bond indices being cut.
        candidate_list (list[int]): A sorted list of all candidate bond indices.
        start (int): The starting index in candidate_list for removal (to prevent duplicates).
        ignore_conjugated (bool): Passed to state checking.

    Returns:
        set[frozenset[int]]: A set of valid states, each represented as a frozenset of bond indices.
    """
    valid_states: set[frozenset[int]] = {frozenset(current_cut_set)}

    for i in range(start, len(candidate_list)):
        bond = candidate_list[i]
        if bond in current_cut_set:
            new_state = current_cut_set.copy()
            new_state.remove(bond)
            if state_is_valid(orig_mol, new_state, ignore_conjugated):
                valid_states.add(frozenset(new_state))
                valid_states.update(_recursive_merge(orig_mol, new_state, candidate_list, i + 1, ignore_conjugated))
    return valid_states

# NOTE: Deprecated
def _random_minimal_fragmentation(
    orig_mol: Chem.Mol, ignore_conjugated: bool = False, num_candidates: int = 1
) -> set[frozenset[int]]:
    """
    Compute one or more valid (minimal) fragmentation states via a greedy randomized procedure.

    Start with all candidate torsional bonds and try to remove bonds in a random order,
    only keeping removals that yield a valid fragmentation (as defined by state_is_valid).

    Args:
        orig_mol (Chem.Mol): The original molecule.
        ignore_conjugated (bool): If True, skip conjugated bonds during torsion detection.
        num_candidates (int): Number of candidate fragmentation states to generate.

    Returns:
        set[frozenset[int]]: A set of valid fragmentation states (each state is a frozenset of bond indices).
    """
    candidates = set()
    valids = set()
    candidate_bonds = list(detect_torsional_bonds(orig_mol, ignore_conjugated))
    
    # Empty Set
    if not candidate_bonds:
        return {frozenset()}

    # Try to generate multiple candidates by shuffling the order of bond removals
    for _ in range(num_candidates):
        current_cut_set = set(candidate_bonds)
        candidate_order = candidate_bonds.copy()
        random.shuffle(candidate_order)

        for bond in candidate_order:
            new_state = current_cut_set.copy()
            new_state.remove(bond)
            if state_is_valid(orig_mol, new_state, ignore_conjugated):
                current_cut_set = new_state
        if state_is_valid(orig_mol, current_cut_set, ignore_conjugated, final_state_check=True):
            valids.add(frozenset(current_cut_set))
        candidates.add(frozenset(current_cut_set))
    if not valids:
        print(f"[WARN] No valid fragmentation reductions found for {Chem.MolToSmiles(orig_mol)}!")
        return candidates
    return valids

# NOTE: Deprecated
def _enumerate_valid_fragmentations(
    orig_mol: Chem.Mol,
    ignore_conjugated: bool = False,
) -> set[frozenset[int]]:
    """
    Enumerate all valid fragmentation patterns.
    A fragmentation pattern is valid if fragmenting using that set of bonds yields
    fragments (of size > 3 atoms) without candidate torsional bonds.

    Args:
        orig_mol (Chem.Mol): The original molecule.
        ignore_conjugated (bool): If True, skip conjugated bonds during torsion detection.

    Returns:
        set[frozenset[int]]: A set of valid bond-cut sets.
    """
    candidate_bonds = detect_torsional_bonds(orig_mol, ignore_conjugated)
    candidate_list = sorted(candidate_bonds)
    if len(candidate_list) == 0:
        return {frozenset()}
    return _recursive_merge(orig_mol, set(candidate_list), candidate_list, 0, ignore_conjugated)


def random_minimal_fragmentation(
    orig_mol: Chem.Mol,
    ignore_conjugated: bool = False,
    num_candidates: int = 1,
    *,
    exclude_ring_bonds: bool = True,
    force_cut_biaryl: bool = False,
    enforce_junction_constraint: bool = True,
    max_passes: int = 1,
) -> set[frozenset[int]]:
    """
    Faster greedy randomized minimal fragmentation. Applied for faster fragmentation during training in large molecules.

    Args:
        orig_mol (Chem.Mol): The original molecule.
        ignore_conjugated (bool): If True, ignore conjugated torsions.
        num_candidates (int): Number of candidates to generate.
        exclude_ring_bonds (bool): If True, exclude ring bonds.
        force_cut_biaryl (bool): If True, force cut biaryl bonds.
        enforce_junction_constraint (bool): If True, enforce the junction constraint.
        max_passes (int): Maximum number of passes.

    Returns:
        set[frozenset[int]]: A set of valid cut sets.
    """

    torsions = sorted(set(detect_torsional_bonds(orig_mol, ignore_conjugated)))
    if not torsions:
        return {frozenset()}

    # --- deterministic pruning (same logic as earlier) ---
    must_cut, removable_candidates, endpoints = deterministic_prune_bonds(
        orig_mol,
        torsions,
        exclude_ring_bonds=exclude_ring_bonds,
        force_cut_biaryl=force_cut_biaryl,
    )

    # Start by cutting everything (same baseline)
    full_cut = frozenset(torsions)

    # Per-call cache of validity
    state_cache: dict[tuple[frozenset[int], bool], bool] = {}

    def is_valid(cut_fs: frozenset[int], final: bool) -> bool:
        key = (cut_fs, final)
        hit = state_cache.get(key)
        if hit is not None:
            return hit
        ok = state_is_valid(orig_mol, cut_fs, ignore_conjugated, final_state_check=final)
        state_cache[key] = ok
        return ok

    # Junction constraint bookkeeping: how many torsion bonds have been UN-CUT at each atom
    # We only need counts for torsion-bond endpoints
    if enforce_junction_constraint:
        uncut_count = [0] * orig_mol.GetNumAtoms()

        def can_uncut(bidx: int) -> bool:
            u, v = endpoints[bidx]
            return uncut_count[u] < 1 and uncut_count[v] < 1

        def apply_uncut(bidx: int) -> None:
            u, v = endpoints[bidx]
            uncut_count[u] += 1
            uncut_count[v] += 1
    else:
        def can_uncut(bidx: int) -> bool:
            return True
        def apply_uncut(bidx: int) -> None:
            return None

    candidates: set[frozenset[int]] = set()
    valids: set[frozenset[int]] = set()

    # Only bonds that are not MUST-CUT are eligible to be removed from the cut set
    removable_candidates = [b for b in removable_candidates if b not in must_cut]

    for _ in range(num_candidates):
        current_cut = set(full_cut)  # start cutting everything
        # reset uncut_count per candidate (only matters if constraint enabled)
        if enforce_junction_constraint:
            for i in range(len(uncut_count)):
                uncut_count[i] = 0

        # random order only over removable candidates
        order = removable_candidates.copy()
        random.shuffle(order)

        # Greedy passes
        for _pass in range(max_passes):
            changed = False
            for b in order:
                if b not in current_cut:
                    continue  # already uncut

                # Cheap feasibility filter before expensive chemistry check
                if not can_uncut(b):
                    continue

                new_cut = frozenset(current_cut - {b})

                # Expensive check only if locally feasible
                if is_valid(new_cut, final=False):
                    current_cut.remove(b)
                    apply_uncut(b)
                    changed = True

            if not changed:
                break  # no further improvements possible

        cut_fs = frozenset(current_cut)
        candidates.add(cut_fs)

        if is_valid(cut_fs, final=True):
            valids.add(cut_fs)

    if not valids:
        print(f"[WARN] No valid fragmentation reductions found for {Chem.MolToSmiles(orig_mol)}!")
        return candidates
    return valids


# ------------------------------------------------------------
# Main enumerator with deterministic pruning + junction constraint
# ------------------------------------------------------------


def enumerate_valid_fragmentations(
    orig_mol: Chem.Mol,
    ignore_conjugated: bool = False,
    *,
    exclude_ring_bonds: bool = True,
    force_cut_biaryl: bool = False,
    enforce_junction_constraint: bool = True,
) -> set[frozenset[int]]:
    """
    Enumerate valid cut sets (frozenset of bond indices) such that all fragments with >3 atoms
    contain no torsional bonds.
    Speedups:
    - Deterministic pruning of MUST-CUT bonds (isolated torsions; optional biaryl)
    - Junction constraint: only consider states where each atom has at most 1 uncut torsion bond, which is a necessary condition for validity and can be tracked cheaply during recursion.
    - Caching of state validity within this call.
    """

    torsions = sorted(set(detect_torsional_bonds(orig_mol, ignore_conjugated)))
    if not torsions:
        return {frozenset()}

    must_cut, removable_candidates, endpoints = deterministic_prune_bonds(
        orig_mol,
        torsions,
        exclude_ring_bonds=exclude_ring_bonds,
        force_cut_biaryl=force_cut_biaryl,
    )

    # Start state: cut everything
    full_cut = frozenset(torsions)

    # Per-call state validity cache (safe with bond indices)
    state_cache: dict[tuple[frozenset[int], bool], bool] = {}

    def is_valid_state(cut_fs: frozenset[int], final: bool) -> bool:
        key = (cut_fs, final)
        hit = state_cache.get(key)
        if hit is not None:
            return hit
        ok = state_is_valid(orig_mol, cut_fs, ignore_conjugated, final_state_check=final)
        state_cache[key] = ok
        return ok

    # For the junction constraint, we track how many torsion-bonds are currently UN-CUT at each atom.
    # Since we start with all torsions cut, uncut_count starts at 0 everywhere and increments when we "remove a cut".
    # Conservative constraint: uncut_count[atom] <= 1
    uncut_count0: tuple[int, ...]
    if enforce_junction_constraint:
        uncut_count0 = tuple([0] * orig_mol.GetNumAtoms())
    else:
        uncut_count0 = tuple()

    # Memoize recursion by (cut_set, start, uncut_count) if constraint enabled
    memo: dict[tuple, set[frozenset[int]]] = {}

    def can_uncut_bond(uncut_count: tuple[int, ...], bidx: int) -> bool:
        if not enforce_junction_constraint:
            return True
        u, v = endpoints[bidx]
        return (uncut_count[u] < 1) and (uncut_count[v] < 1)

    def apply_uncut(uncut_count: tuple[int, ...], bidx: int) -> tuple[int, ...]:
        if not enforce_junction_constraint:
            return uncut_count
        u, v = endpoints[bidx]
        lst = list(uncut_count)
        lst[u] += 1
        lst[v] += 1
        return tuple(lst)

    def recursive_merge(
        current_cut: frozenset[int],
        start: int,
        uncut_count: tuple[int, ...],
    ) -> set[frozenset[int]]:
        key = (current_cut, start, uncut_count) if enforce_junction_constraint else (current_cut, start)
        if key in memo:
            return memo[key]

        valid_states: set[frozenset[int]] = {current_cut}

        # Only try to uncut bonds that are NOT in must_cut
        for i in range(start, len(removable_candidates)):
            b = removable_candidates[i]

            if b in must_cut:
                continue
            if b not in current_cut:
                continue
            if not can_uncut_bond(uncut_count, b):
                continue

            new_cut = current_cut - {b}
            new_uncut = apply_uncut(uncut_count, b)

            # Chemistry-ground-truth check
            if is_valid_state(new_cut, final=False):
                valid_states.add(new_cut)
                valid_states.update(recursive_merge(new_cut, i + 1, new_uncut))

        memo[key] = valid_states
        return valid_states

    # IMPORTANT: if must_cut is non-empty, it's already included in full_cut (since full_cut is all torsions).
    return recursive_merge(full_cut, 0, uncut_count0)


# ---------------------------------------------------------------
# 6. Merging Trivial Fragments
# ---------------------------------------------------------------


def select_cut_sets_with_crirerion(
    mol: Chem.Mol,
    cut_sets: list[frozenset[int]],
    indices: list[int],
    criterion: Literal["largest", "smallest"] = "largest",
) -> list[frozenset[int]]:
    """
    From the minimal cut sets, selects all that yield the maximum average
    heavy atom count per fragment.

    Args:
        mol (Mol): The molecule to fragment.
        cut_sets (List[FrozenSet[int]]): All valid cut sets sorted by size.
        indices (List[int]): Indices of the minimal cut sets to consider.
        criterion (Literal["largest", "smallest"]): The criterion for selection.

    Returns:
        List[FrozenSet[int]]: A list of cut sets with the highest average fragment size.
    """
    best_avg = -1.0
    tol = 1e-5
    best_cuts = []

    def _count_heavy_atoms(mol: Chem.Mol) -> int:
        """Count the number of heavy (non-hydrogen) atoms in the molecule."""
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

    for idx in indices:
        candidate = cut_sets[idx]
        frag = fragment_on_bonds_with_mapping(mol, list(candidate))
        frags = get_fragments_as_mols(frag)
        if criterion == "largest":
            score = max(_count_heavy_atoms(f) ** 2 for f in frags)
        else:
            score = sum(_count_heavy_atoms(f) ** 0.5 for f in frags) / len(frags)

        if score > best_avg + tol:
            best_avg = score
            best_cuts = [candidate]
        elif abs(score - best_avg) <= tol:
            best_cuts.append(candidate)

    return best_cuts


def fragment_molecule(  # noqa
    mol: Chem.Mol,
    selection: Literal["random", "random_all", "max", "largest", "smallest", "canonical"] = "random",
    ignore_conjugated: bool = False,
    verbose: bool = False,
    max_recursive: int = 12,
) -> Chem.Mol:
    """
    Fragment a molecule using a selected minimal cut set strategy.

    This function first enumerates all valid bond-cut sets using
    `enumerate_valid_fragmentations(mol, ignore_conjugated=False)`. It then identifies
    those cut sets that result in the minimal number of bonds being cut.

    Based on the selection mode:

      - "random": Recommended for training. Randomly sample one from the minimal cut sets.
      - "random_all": Randomly sample one from all valid cut sets.
      - "max": (NOT RECOMMENDED) Select the one with the most number fragments (skip merge)
      - "largest": Select the one with the largest fragments
      - "smallest": Select from the minimal cut sets, the one that yields the largest
        average fragment size (evaluated by the average sqrt(heavy atom count) across fragments),
        which prioritizes merging small fragments.
      - "canonical": (Not implemented yet) Would define a canonical minimal ordering,
        e.g. maximizing merging from leaf node to center of rotation recursively.

    Args:
        mol (Mol): The RDKit molecule to fragment.
        selection (Literal["random", "largest", "smallest", "canonical"]): The selection strategy to use.
        ignore_conjugated (bool): If True, skip conjugated bonds during torsion detection.
        This is passed to `enumerate_valid_fragmentations`.

    Returns:
        Mol: A new RDKit molecule fragmented at the chosen bond-cut set.
             If no valid fragmentations are found, returns the input molecule unchanged.
    """

    def _get_chiral_bonds(mol: Chem.Mol) -> set[int]:
        """
        Identifies the indices of all bonds connected to a chiral center.
        """
        # Find all chiral centers, including those that are not explicitly assigned (R/S)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        chiral_atom_indices = {center[0] for center in chiral_centers}

        bonds_to_preserve = set()
        for atom_idx in chiral_atom_indices:
            atom = mol.GetAtomWithIdx(atom_idx)
            for bond in atom.GetBonds():
                # Store the index of any bond connected to this chiral atom
                bonds_to_preserve.add(bond.GetIdx())

        return bonds_to_preserve

    all_torsionals = sorted(detect_torsional_bonds(mol, ignore_conjugated))
    if len(all_torsionals) == 0:
        # No torsional bonds detected, return the original molecule (func will add empty mapping)
        return fragment_on_bonds_with_mapping(mol, [], ignore_conjugated=ignore_conjugated)

    if selection == "max":
        # Vanilla fragmentation with largest number of fragments
        if verbose:
            print("[WARN] 'most' selection is not recommended for training.")
        chosen_cut_set = all_torsionals
    elif len(all_torsionals) >= max_recursive:
        if verbose:
            print(f"[WARN] Using random minimal fragmentation for {Chem.MolToSmiles(mol)}. Too many torsional bonds.")
        # Recommended for training
        valid_cuts = list(
            random_minimal_fragmentation(
                mol,
                ignore_conjugated=ignore_conjugated,
                num_candidates=min(max_recursive, 2 + len(all_torsionals) // 2),
            )
        )
        ascending_valid_cuts = sorted(valid_cuts, key=lambda s: (len(s), sorted(s)))
        # NOTE in this case we sample randomly from all minimal cut sets
        num_cuts = [len(cut_set) for cut_set in ascending_valid_cuts]
        min_cut_size = min(num_cuts)
        # Allow leeway of 1 cut to avoid over-determining the fragmentation
        minimal_indices = np.where(np.array(num_cuts) <= min_cut_size)[0]
        chosen_cut_set = random.choice([ascending_valid_cuts[i] for i in minimal_indices])
    else:
        valid_fragmentations = list(enumerate_valid_fragmentations(mol, ignore_conjugated=ignore_conjugated))
        if not valid_fragmentations:
            # No valid fragmentations found, return the original molecule
            print(f"[WARN] No valid fragmentations found for {Chem.MolToSmiles(mol)}. Not fragmenting...\n")
            return mol

        # Sort by number of cuts (then lexicographically to ensure consistent ordering)
        ascending_valid_cuts = sorted(valid_fragmentations, key=lambda s: (len(s), sorted(s)))
        num_cuts = [len(cut_set) for cut_set in ascending_valid_cuts]
        min_cut_size = min(num_cuts)

        # Get indices of all minimal cut sets
        # Allow leeway of 1 cut to avoid over-determining the fragmentation
        minimal_indices = np.where(np.array(num_cuts) <= min_cut_size)[0]
        minimal_cut_sets = [ascending_valid_cuts[i] for i in minimal_indices]

        if selection == "random":
            # Recommended for training. Random minimal fragmentation cuts.
            chosen_cut_set = random.choice(minimal_cut_sets)
        elif selection == "random_all":
            chosen_cut_set = random.choice(ascending_valid_cuts)
        elif selection == "smallest":
            # Good for inference. Prioritizes smaller fragment merging.
            optimal_cuts = select_cut_sets_with_crirerion(
                mol,
                ascending_valid_cuts,
                minimal_indices.tolist(),
                criterion="smallest",
            )
            chosen_cut_set = random.choice(optimal_cuts)
        elif selection == "largest":
            optimal_cuts = select_cut_sets_with_crirerion(
                mol, ascending_valid_cuts, minimal_indices.tolist(), criterion="largest"
            )
            chosen_cut_set = random.choice(optimal_cuts)
        elif selection == "canonical":
            # Best for inference
            # NOTE some canonical merge would prioritize i.e. merging leaf nodes first... COOH groups together, CONH...
            # NOTE can also attempt to minimize the consecutive length of fragments instead of SIZE
            # NOTE could also use TorsionFingerprints Weighting to iterate through leafs first!
            # Identify all bonds connected to chiral centers
            bonds_to_preserve = _get_chiral_bonds(mol)

            # Filter the minimal cut sets to find those that are "safe"
            chirality_preserving_cuts = []
            for cut_set in minimal_cut_sets:
                # A cut set is "safe" if it has no intersection with the bonds to preserve
                if not bonds_to_preserve.intersection(cut_set):
                    chirality_preserving_cuts.append(cut_set)

            if chirality_preserving_cuts:
                # If we found safe cuts, choose one randomly from that pool
                if verbose:
                    print("[INFO] Found chirality-preserving fragmentation.")
                chosen_cut_set = random.choice(chirality_preserving_cuts)
            else:
                # Fallback: if no minimal cut can preserve chirality, print a warning
                # and revert to the standard "random" minimal selection.
                if verbose:
                    print(
                        f"[WARN] No minimal fragmentation found for {Chem.MolToSmiles(mol)} that preserves \
                        all chiral centers. Using random minimal cut."
                    )
                chosen_cut_set = random.choice(minimal_cut_sets)
        else:
            raise ValueError(
                f"Invalid selection mode: {selection}. Choose from 'random', 'largest', 'smallest', or 'canonical'."
            )

    # Fragment the molecule at the selected bond indices
    return fragment_on_bonds_with_mapping(mol, list(chosen_cut_set), ignore_conjugated=ignore_conjugated)


def get_fragmented_anchors_dummies(mol: Chem.Mol) -> tuple[list[int], list[int]]:
    """Get the indices of the anchors and dummies in a fragmented molecule.
    The anchors are the heavy atoms (atomic number > 0) that are connected to dummy atoms (atomic number 0).

    arg:
        mol (Chem.Mol): The fragmented molecule.
    Returns:
        tuple[list[int], list[int]]: A tuple containing two lists:
            - anchors: The indices of the anchor atoms.
            - dummies: The indices of the dummy atoms.
    """
    anchors, dummies = [], []
    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtom(), bond.GetEndAtom()
        # Get non-repeating anchor-dummy indices
        if dst.GetAtomicNum() == 0:
            # print(src.GetProp("anchor_idx"), dst.GetProp("anchor_idx"))
            assert src.GetAtomicNum() > 0
            anchors.append(src.GetIdx())
            dummies.append(dst.GetIdx())
        elif src.GetAtomicNum() == 0:
            # print(src.GetProp("dummy_idx"), dst.GetProp("dummy_idx"))
            assert dst.GetAtomicNum() > 0
            anchors.append(dst.GetIdx())
            dummies.append(src.GetIdx())
    # Checks for consistency, should always be true
    assert sum([a == "*" for a in Chem.MolToSmiles(mol)]) == len(anchors), "Number of anchors and dummies do not match"
    assert len(anchors) == len(dummies), "Number of anchors and dummies do not match"
    return anchors, dummies


# -----------------------------------------------------------------
# 6.5 Virtual Node Construction
# -----------------------------------------------------------------


def get_ring_centers_with_atoms(mol: Chem.Mol, atom_mask: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    """
    Return a list of tuples (center, ring_atoms) for each ring in the molecule,
    excluding any rings that contain masked-off atoms.

    Args:
        mol (Chem.Mol): Molecule with 3D conformer.
        atom_mask (np.ndarray): Boolean mask of shape (num_atoms,) where True indicates
            atoms to include, False indicates atoms to ignore.

    Returns:
        List of (center, ring_atoms) where ring_atoms is a list of atom indices.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule must have 3D coordinates.")
    conf = mol.GetConformer()
    ring_info = mol.GetRingInfo()
    rings = []
    for ring in ring_info.AtomRings():
        # skip rings containing any masked atom
        if any(not atom_mask[idx] for idx in ring):
            continue
        coords = []
        for idx in ring:
            pos = conf.GetAtomPosition(idx)
            coords.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))
        center = np.stack(coords, axis=0).mean(axis=0)
        rings.append((center, list(ring)))
    return rings


def get_all_virtual_nodes_per_fragment(  # noqa
    mol: Chem.Mol,
    fragment_ids: list[list[int]],
    atom_mask: np.ndarray | None = None,
) -> dict[int, list[dict]]:
    """
    For each fragment (list of atom indices) in the molecule, create virtual node(s),
    using ring centers when available and fragment CoM otherwise, while respecting atom_mask.

    Args:
        mol (Chem.Mol): Molecule with a 3D conformer.
        fragment_ids (list[list[int]]): List of fragments, each a list of atom indices.
        atom_mask (np.ndarray): Boolean mask (shape: num_atoms) indicating which atoms to include.

    Returns:
        dict mapping fragment index to a list of virtual node dicts with keys:
          - "type": "ring" or "CoM"
          - "coords": np.ndarray of shape (3,)
          - "connected_atoms": list of atom indices (within fragment)
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("The molecule must have 3D coordinates.")
    conf = mol.GetConformer()

    if atom_mask is None:
        atom_mask = np.array([True] * mol.GetNumAtoms())
    elif len(atom_mask) != mol.GetNumAtoms():
        raise ValueError(f"atom_mask must be of shape ({mol.GetNumAtoms()},)")
    atom_mask = np.array(atom_mask, dtype=bool)

    # Precompute ring centers excluding masked atoms
    rings_full = get_ring_centers_with_atoms(mol, atom_mask)

    frag_virtual_nodes: dict[int, list[dict]] = {}

    for frag_idx, frag_atom_idxs in enumerate(fragment_ids):
        # restrict fragment atoms by mask
        valid_atoms = [idx for idx in frag_atom_idxs if atom_mask[idx]]
        virtual_nodes: list[dict] = []

        # find rings wholly contained in the fragment
        rings_in_frag = []
        for center, ring_atoms in rings_full:
            if set(ring_atoms).issubset(valid_atoms):
                rings_in_frag.append((center, ring_atoms))

        if rings_in_frag:
            # create one virtual node per ring
            for center, ring_atoms in rings_in_frag:
                virtual_nodes.append({"type": "ring", "coords": center, "connected_atoms": ring_atoms})
        else:
            # no rings: compute CoM of valid_atoms
            coords_list = []
            masses = []
            for idx in valid_atoms:
                pos = conf.GetAtomPosition(idx)
                coords_list.append(np.array([pos.x, pos.y, pos.z], dtype=np.float32))
                atom = mol.GetAtomWithIdx(idx)
                mass = atom.GetMass() if atom.GetMass() > 0 else 12.0
                masses.append(mass)
            if not coords_list:
                # fragment has no valid atoms after masking
                frag_virtual_nodes[frag_idx] = []
                continue
            coords_arr = np.stack(coords_list, axis=0)
            masses_arr = np.array(masses, dtype=np.float32)
            com = np.average(coords_arr, axis=0, weights=masses_arr)
            virtual_nodes.append({"type": "CoM", "coords": com, "connected_atoms": valid_atoms})

        frag_virtual_nodes[frag_idx] = virtual_nodes

    return frag_virtual_nodes


# ---------------------------------------------------------------
# 7. Visualization Functions
# ---------------------------------------------------------------


def get_fragments_as_mols(mol: Chem.Mol, asMols: bool = True) -> list[Chem.Mol | tuple[list[int], list[int]]]:
    """
    Return the connected components (fragments) of a molecule as a list of Mol objects or tuples of atom indices.

    Parameters:
        mol (Chem.Mol): The molecule.
        asMols (bool): If True, return the fragments as Mol objects.

    Returns:
        list[Chem.Mol]: A list of fragment molecules.
    """
    # Making it explicit for visual.
    if asMols:
        return Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=True)


def visualize_fragments(mol: Chem.Mol, mols_per_row: int = 4, sub_img_size: tuple[int, int] = (250, 250)) -> Image:
    """
    Visualize the fragments of a molecule in a grid.

    Parameters:
        mol (Chem.Mol): The molecule.
        mols_per_row (int): Number of fragments per row in the grid.
        sub_img_size (tuple[int, int]): Size of each fragment image.

    Returns:
        Image: A PIL Image showing the fragments in a grid.
    """
    frags = get_fragments_as_mols(mol)
    legends = [f"Fragment {i + 1}" for i in range(len(frags))]
    img = Draw.MolsToGridImage(frags, molsPerRow=mols_per_row, subImgSize=sub_img_size, legends=legends)
    return img


def visualize_fragments_on_molecule(
    original_mol: Chem.Mol,
    fragments: list[Chem.Mol],
    img_size: tuple[int, int] = (600, 600),
    legend: str = "Fragments on Molecule",
) -> Image:
    """
    Visualize fragments by highlighting them on the original molecule.

    Parameters:
        original_mol (Chem.Mol): The original molecule.
        fragments (list[Chem.Mol]): List of fragment molecules.
        img_size (tuple[int, int]): Size of the output image.
        legend (str): Legend to display below the molecule.

    Returns:
        Image: A PIL Image showing the molecule with fragments highlighted.
    """
    if not original_mol.GetNumConformers():
        AllChem.Compute2DCoords(original_mol)

    colors = [
        (0.8, 0.2, 0.2),
        (0.2, 0.8, 0.2),
        (0.2, 0.2, 0.8),
        (0.8, 0.8, 0.2),
        (0.8, 0.2, 0.8),
        (0.2, 0.8, 0.8),
        (0.6, 0.4, 0.2),
        (0.4, 0.6, 0.4),
        (0.4, 0.4, 0.6),
    ]
    atom_colors = {}
    assigned_atoms = set()

    for i, frag in enumerate(fragments):
        # Remove dummy atoms for matching.
        clean_frag = Chem.RWMol(frag)
        dummy_indices = [atom.GetIdx() for atom in clean_frag.GetAtoms() if atom.GetAtomicNum() == 0]
        for idx in sorted(dummy_indices, reverse=True):
            clean_frag.RemoveAtom(idx)
        clean_frag = clean_frag.GetMol()

        matches = original_mol.GetSubstructMatches(clean_frag, useChirality=True, uniquify=True)
        if matches:
            best_match = None
            min_overlap = float("inf")
            for match in matches:
                match_set = set(match)
                overlap = len(match_set.intersection(assigned_atoms))
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_match = match
            if best_match:
                color_idx = i % len(colors)
                for atom_idx in best_match:
                    atom_colors[atom_idx] = colors[color_idx]
                    assigned_atoms.add(atom_idx)

    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(*img_size)
    drawer.DrawMolecule(
        original_mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        legend=legend,
    )
    drawer.FinishDrawing()

    png_data = drawer.GetDrawingText()
    import io

    image = Image.open(io.BytesIO(png_data))
    return image


def display_valid_fragment_images(
    orig_mol: Chem.Mol,
    valid_fragmentations: Seq[set[int]],
    indices: list[int],
    visualize_fragments_on_molecule: Callable[[Chem.Mol, list[Chem.Mol]], Image.Image],
    get_fragments_as_mols: Callable[[Chem.Mol], list[Chem.Mol]],
    size_factor: float = 1.0,
) -> None:
    """
    Display several fragmentation visualizations in a grid.

    Parameters:
        orig_mol (Chem.Mol): The original molecule.
        valid_fragmentations (Sequence[set[int]]): A sorted sequence of valid cut sets.
        indices (list[int]): List of indices into valid_fragmentations to visualize.
        visualize_fragments_on_molecule (Callable): A function that takes (orig_mol, fragments)
            and returns an image.
        get_fragments_as_mols (Callable): A function that takes a fragmented molecule and returns
            a list of fragment molecules.
    """
    images: list[Image.Image] = []

    for idx in indices:
        cut_set = list(valid_fragmentations[idx])
        frag_mol = fragment_on_bonds(orig_mol, cut_set)
        frag_mols = get_fragments_as_mols(frag_mol)
        img = visualize_fragments_on_molecule(orig_mol, frag_mols)
        images.append(img)

    n_images = len(images)
    ncols = math.ceil(math.sqrt(n_images))
    nrows = math.ceil(n_images / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols * size_factor, 3 * nrows * size_factor))
    axes = axes.flatten() if n_images > 1 else [axes]

    for ax, img, idx in zip(axes, images, indices):
        ax.imshow(img)
        ax.axis("off")
        num_frags = len(get_fragments_as_mols(fragment_on_bonds(orig_mol, list(valid_fragmentations[idx]))))
        ax.set_title(f"Num Frags: {num_frags}")

    for ax in axes[len(images) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
# 8. I/O Utility
# ---------------------------------------------------------------


def save_fragments_to_sdf(fragments: list[Chem.Mol], filename: Union[str, Path]) -> None:
    """
    Save a list of fragment molecules to an SDF file.

    Parameters:
        fragments (list[Chem.Mol]): The fragment molecules.
        filename (str | Path): Output file path.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(filename))
    for frag in fragments:
        writer.write(frag)
    writer.close()


# ---------------------------------------------------------------
# Example Usage (if run as main)
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Example molecule; replace with any molecule's SMILES.
    smiles = "CC(C)CC1=CC=CC=C1"
    orig_mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(orig_mol)

    # Full fragmentation: using all candidate bonds.
    full_cut_bonds = sorted(detect_torsional_bonds(orig_mol, ignore_conjugated=False))
    frag_all = fragment_on_bonds_with_mapping(orig_mol, full_cut_bonds)
    print("Full fragmentation (all cuts):")
    print(Chem.MolToSmiles(frag_all))

    # Enumerate valid fragmentation patterns.
    valid_fragmentations = enumerate_valid_fragmentations(orig_mol, ignore_conjugated=False)
    print("\nValid fragmentation patterns (each as sorted list of cut indices):")
    for state in sorted(valid_fragmentations, key=lambda s: (len(s), sorted(s))):
        print(sorted(state))

    # Dummy visualization functions for demonstration:
    def my_visualize_fragments_on_molecule(mol: Chem.Mol, frags: list[Chem.Mol]) -> Image.Image:
        # Use RDKit's drawing to generate an image.
        return Draw.MolToImage(mol, size=(300, 300))

    def my_get_fragments_as_mols(mol: Chem.Mol) -> list[Chem.Mol]:
        return Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)

    ascending_valid_cuts = sorted(valid_fragmentations, key=lambda s: (len(s), sorted(s)))
    indices_to_display = [0, 1, 2, 3, 4, 5]
    display_valid_fragment_images(
        orig_mol,
        ascending_valid_cuts,
        indices_to_display,
        my_visualize_fragments_on_molecule,
        my_get_fragments_as_mols,
    )
