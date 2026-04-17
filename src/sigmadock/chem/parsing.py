import os
import warnings
from io import StringIO
from pathlib import Path
from typing import Union

import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO, PDBParser, Select
from rdkit import Chem
from rdkit.Chem.rdchem import AtomValenceException, KekulizeException
from rdkit.rdBase import BlockLogs
from scipy.spatial import cKDTree

from sigmadock.chem.processing import get_coordinates

warnings.filterwarnings("ignore")

# -----------------------------------------------------
# ---------------------- GENERAL ----------------------
# -----------------------------------------------------


def compute_com(mol: Chem.Mol, heavy_atoms_only: bool = True, weighted: bool = False) -> np.ndarray:
    """Compute the center of mass (COM) of a molecule."""
    assert mol.GetNumConformers() > 0, "Molecule has no conformers."
    if mol.GetNumConformers() > 1:
        warnings.warn("Molecule has multiple conformers. Using the first one.", stacklevel=2)

    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()])
    if heavy_atoms_only:
        coords = coords[[atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()]]
    if coords.size == 0:
        raise ValueError("No heavy atoms found in the molecule.")
    if weighted:
        weights = np.array([atom.GetMass() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1])
        com = np.average(coords, axis=0, weights=weights)
    else:
        com = coords.mean(axis=0)
    return com


# -----------------------------------------------------
# ---------------------- LIGAND ----------------------
# -----------------------------------------------------


def __read_ligands_from_sdf(ligand_sdf: str | Path) -> list[Chem.Mol]:
    """Read ligands from an SDF file and return RDKit Mol objects."""
    ligand_supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=True)
    ligand_mols = [mol for mol in ligand_supplier if mol is not None]

    if not ligand_mols:
        raise ValueError(f"No valid ligands found in SDF file: {ligand_sdf}")

    return ligand_mols


def _try_sanitize(mol: Chem.Mol) -> Chem.Mol | None:
    """
    Try to fully sanitize the molecule. On kekulization or valence errors,
    fall back to writing a MolBlock without kekulization and reparsing.
    If that also fails, return None.
    """
    blocker = BlockLogs()
    try:
        # First attempt at full sanitization
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        return mol
    except (KekulizeException, AtomValenceException):
        # Fallback: dump without kekulization, then re-parse
        try:
            block = Chem.rdmolfiles.MolToMolBlock(mol, kekulize=False)
            fixed = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
            return fixed
        except Exception:
            return None
    finally:
        # This guarantees logs are always re-enabled, no matter which return path is taken
        del blocker


def _read_sdf(path: Union[str, Path]) -> list[Chem.Mol]:
    """Load raw molecules from an SDF, no sanitization or H removal."""
    block_logs = BlockLogs()
    try:
        mols = [m for m in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False) if m]
    finally:
        # This 'finally' block ensures logs are ALWAYS re-enabled
        del block_logs
    return mols


def _read_mol2(path: Path) -> list[Chem.Mol]:
    """Load (single-mol) Mol2 via MolFromMol2File, no supplier exists."""
    m = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    return [m] if m else []

def write_sdf(mols: list[Chem.Mol], path: Union[str, Path]) -> None:
    """Write a list of RDKit Mol objects to an SDF file."""
    writer = Chem.SDWriter(str(path))
    for mol in mols:
        writer.write(mol)
    writer.close()

def read_ligands_from_sdf(ligand_sdf: Union[str, Path], remove_hs: bool = True) -> list[Chem.Mol]:
    """
    Read ligands from an SDF (stepwise sanitize each), with a .mol2 fallback.

    1) Load raw with SDMolSupplier (sanitize=False)
    2) Sanitize each via _try_sanitize
    3) If none survive, warn + try .mol2 next to the .sdf
    4) Optionally RemoveHs on all survivors
    5) Raise if still empty
    """
    path = Path(ligand_sdf)
    mols: list[Chem.Mol] = []

    # Step 1 & 2: SDF load + sanitize
    for raw in _read_sdf(path):
        fixed = _try_sanitize(raw)
        if fixed:
            mols.append(fixed)

    # Step 3: fallback to Mol2 if still empty
    if not mols:
        mol2_path = path.with_suffix(".mol2")
        if mol2_path.exists():
            warnings.warn(f"No valid SDF ligands in {path.name}; trying Mol2 {mol2_path.name}", stacklevel=2)
            for raw in _read_mol2(mol2_path):
                fixed = _try_sanitize(raw)
                if fixed:
                    mols.append(fixed)

    if not mols:
        raise ValueError(f"No valid, sanitized ligands found in {path} or its .mol2 counterpart")

    # Step 4: strip explicit Hs if desired
    if remove_hs:
        mols = [Chem.RemoveHs(m) for m in mols]

    return mols


# def read_ligands_from_sdf(
#     ligand_sdf: Path | str, remove_hs: bool = True
# ) -> list[Chem.Mol]:
#     """
#     Read ligands from an SDF file, sanitizing each one individually.

#     Args:
#         ligand_sdf: path to .sdf (or .mol2 fallback)
#         remove_hs: whether to strip explicit Hs after reading

#     Returns:
#         List of sanitized RDKit Mol objects.
#     """
#     ligand_sdf = str(ligand_sdf)

#     # 1) Try SDF supplier without auto-sanitization
#     supplier = Chem.SDMolSupplier(ligand_sdf, sanitize=False, removeHs=False)
#     mols = []

#     for raw in supplier:
#         if raw is None:
#             continue
#         # 2) Try to sanitize (includes kekulize, valence checks…)
#         try:
#             Chem.SanitizeMol(
#                 raw,
#                 sanitizeOps=(
#                     Chem.SanitizeFlags.SANITIZE_ALL
#                     # & ~Chem.SanitizeFlags.SANITIZE_CLEANUP  # optional: skip cleanup
#                 ),
#             )
#         except Exception:
#             # 3) Fallback: re-parse the molblock manually
#             block = Chem.rdmolfiles.MolToMolBlock(raw, kekulize=False)
#             try:
#                 fixed = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
#                 if fixed is not None:
#                     raw = fixed
#                 else:
#                     # if still failing, skip
#                     continue
#             except Exception:
#                 continue

#         mols.append(raw)

#     # 4) If we got nothing, try Mol2
#     if not mols:
#         lig_path = Path(ligand_sdf)
#         mol2_path = lig_path.with_suffix(".mol2")
#         if mol2_path.exists():
#             m2 = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
#             if m2 is not None:
#                 try:
#                     Chem.SanitizeMol(m2)
#                     mols.append(m2)
#                 except Exception as e:
#                     raise ValueError(f"Could not read any molecules from {ligand_sdf}. Error: {e}") from e

#     # 5) Finally remove Hs if requested
#     if remove_hs:
#         mols = [Chem.RemoveHs(m) for m in mols]

#     if not mols:
#         raise ValueError(f"No valid, sanitized ligands found in {ligand_sdf}")

#     return mols


# -----------------------------------------------------
# ---------------------- PROTEIN ----------------------
# -----------------------------------------------------


def compute_residue_centroid(residue: PDB.Residue.Residue) -> np.ndarray:
    coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
    return coords.mean(axis=0) if coords.shape[0] > 0 else None


def filter_outlier_residues(
    residue_ids: list[tuple[str, int]],
    structure: PDB.Structure,
    std_factor: float = 1.5,
    keep_dist: float = 8.0,
) -> list[bool]:
    """
    Filters out residues that are too far from the overall centroid of the residues.
    Returns a list of boolean values indicating whether each residue is valid after filtering.
    """
    centroids = []
    valid_residues = []

    # Loop through the residues and compute centroids
    for chain_id, res_id in residue_ids:
        chain = structure[0][chain_id]
        res = chain[res_id]
        cent = compute_residue_centroid(res)
        if cent is not None:
            centroids.append(cent)
            valid_residues.append((chain_id, res_id))

    # Calculate overall centroid and distances
    centroids_arr = np.array(centroids)
    overall_centroid = centroids_arr.mean(axis=0)
    distances = np.linalg.norm(centroids_arr - overall_centroid, axis=1)

    # Calculate mean distance, std deviation, and cutoff
    mean_dist = distances.mean()
    std_dist = distances.std()
    cutoff = mean_dist + std_factor * std_dist

    # Return boolean array for which residues are within the cutoff
    return [dist <= cutoff or dist <= keep_dist for dist in distances]


def split_protein_by_chain(
    protein_pdb: str | Path,
    output_dir: Path | str | None = None,
    select_chains: list[str] | None = None,
    return_as_string: bool = False,
) -> dict[str, str]:
    """
    Splits the protein into individual chains and either saves them to files or returns PDB strings.

    Args:
        protein_pdb (str | Path): Path to the protein PDB file.
        output_dir (Path | str | None): Directory to save chain PDBs. Ignored if return_as_string=True.
        select_chains (list[str] | None): List of chain IDs to extract. If None, extracts all chains.
        return_as_string (bool): If True, returns PDB strings instead of writing files.

    Returns:
        dict[str, str]: Dictionary mapping chain IDs to PDB strings (if return_as_string=True).
    """
    protein_pdb = str(protein_pdb)
    if not os.path.isfile(protein_pdb):
        raise FileNotFoundError(f"File {protein_pdb} does not exist.")
    if not protein_pdb.endswith(".pdb"):
        raise ValueError(f"File {protein_pdb} is not a PDB file.")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_pdb)
    protein_id = os.path.basename(protein_pdb).split("_")[0]

    chain_data = {}

    class ChainSelect(Select):
        """Biopython select class to save a specific chain."""

        def __init__(self, target_chain: str) -> None:
            self.target_chain = target_chain

        def accept_chain(self, chain: PDB.Chain) -> bool:
            return chain.id == self.target_chain

    for model in structure:
        for chain in model:
            chain_id = chain.id

            if select_chains and chain_id not in select_chains:
                continue

            io = PDBIO()
            io.set_structure(structure)

            if return_as_string:
                pdb_buffer = StringIO()
                io.save(pdb_buffer, select=ChainSelect(chain_id))
                chain_data[chain_id] = pdb_buffer.getvalue()
            else:
                output_dir = output_dir or os.path.dirname(protein_pdb)
                os.makedirs(output_dir, exist_ok=True)

                out_path = os.path.join(output_dir, f"chains/{protein_id}_{chain_id}.pdb")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                io.save(out_path, select=ChainSelect(chain_id))

    return chain_data if return_as_string else {}


def prune_structure(
    struct: PDB.Structure,
    remove_waters: bool = True,
    remove_hetatoms: bool = True,
    remove_hydrogens: bool = True,
) -> None:
    """
    Mutate `struct` in place, removing:
      - water residues (resname HOH/WAT) if remove_waters
      - any HETATM residues (residue.id[0] != ' ') if remove_hetatoms
      - any atom whose element is H (explicit hydrogens) if remove_hydrogens
    """
    for model in struct:
        for chain in model:
            # first: collect residues to drop
            drop_res = []
            for residue in list(chain):
                het_flag = residue.id[0]  # ' ' = standard AA, anything else = HET
                name = residue.get_resname().strip()

                if (remove_waters and name in ("HOH", "WAT")) or (remove_hetatoms and het_flag != " "):
                    drop_res.append(residue.id)

            # drop them
            for res_id in drop_res:
                chain.detach_child(res_id)

            # now within each kept residue drop H atoms
            if remove_hydrogens:
                for residue in chain:
                    for atom in list(residue):
                        elem = atom.element.strip() if atom.element else atom.get_name()[0]
                        if elem.upper() == "H":
                            residue.detach_child(atom.get_id())


def get_protein(
    protein_pdb: str | Path,
    remove_waters: bool = True,
    remove_hetatoms: bool = True,
    remove_hydrogens: bool = True,
) -> PDB.Structure:
    """Load a protein structure, then optionally strip waters, HETATMs, and H-atoms."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("protein", str(protein_pdb))

    prune_structure(
        struct,
        remove_waters=remove_waters,
        remove_hetatoms=remove_hetatoms,
        remove_hydrogens=remove_hydrogens,
    )
    return struct


def get_protein_chain(protein_pdb: str | Path, chain_id: str) -> PDB.Chain:
    """Load a specific chain from a protein structure."""
    structure = get_protein(protein_pdb)
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return chain
    raise ValueError(f"Chain {chain_id} not found in {protein_pdb}.")


def structure_to_rdkit(structure: PDB.Structure, remove_hs: bool = False) -> Chem.Mol:
    """
    Convert a BioPython PDB Structure into an RDKit Mol by:
      1. Writing it to an in-memory PDB block
      2. Feeding that block to RDKit's MolFromPDBBlock

    Args:
        structure: a Bio.PDB.Structure instance
        remove_hs: whether to drop explicit hydrogens in the RDKit Mol

    Returns:
        An RDKit Mol with 3D coordinates from the original Structure.
    """
    # 1) Dump structure to a PDB format string
    io = PDBIO()
    io.set_structure(structure)
    buffer = StringIO()
    io.save(buffer)
    pdb_block = buffer.getvalue()

    # 2) Parse via RDKit
    mol = Chem.MolFromPDBBlock(
        pdb_block,
        removeHs=remove_hs,
        sanitize=True,
    )
    if mol is None:
        raise ValueError("RDKit failed to parse the PDB block")
    return mol


def inspect_structure(structure: PDB.Structure) -> tuple[bool, bool, bool]:
    """
    Scan a Bio.PDB Structure and return:
      (has_waters, has_explicit_hydrogens, has_hetatoms)

    - has_waters is True if any residue named “HOH” or “WAT” is present.
    - has_explicit_hydrogens is True if any atom whose element is H
      (or atom name starts with 'H') is present.
    - has_hetatoms is True if any residue has a non-blank hetero-flag
      (residue.id[0] != ' '), excluding water.
    """
    has_waters = False
    has_h = False
    has_hetatoms = False

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                het_flag = residue.id[0]  # ' ' = standard AA; anything else = HETATM

                # 1) Water check
                if resname in ("HOH", "WAT"):
                    has_waters = True

                # 2) HETATM check (non-standard residues, excluding water)
                if het_flag != " " and resname not in ("HOH", "WAT"):
                    has_hetatoms = True

                # 3) Explicit H atom check
                for atom in residue:
                    elem = atom.element.strip() if atom.element else atom.get_name()[0]
                    if elem.upper() == "H":
                        has_h = True

                # Early exit if all three flags are True
                if has_waters and has_h and has_hetatoms:
                    return has_waters, has_h, has_hetatoms

    return has_waters, has_h, has_hetatoms


def get_protein_coordinates(protein_pdb: str | Path) -> np.ndarray:
    """Return (N,3) NumPy array of protein atom (x, y, z) coordinates."""
    structure = get_protein(protein_pdb)
    coords = np.array(
        [atom.get_coord() for model in structure for chain in model for residue in chain for atom in residue]
    )
    return coords


def extract_pockets_deprecated(  # noqa
    protein_pdb: str | Path,
    ligand_sdf: str | Path,
    distance_cutoff: float = 10.0,
    keep_hetatoms: bool = True,
    return_as_string: bool = True,
    output_dir: str | None = None,
    noise: float = 0.5,
    filter_outlier_factor: float = -1,
    random_sample: bool = False,
) -> tuple[list[list[tuple[str, int]]], list[Chem.Mol]]:
    """
    Extracts binding pockets from a protein PDB file using ligand coordinates.
    Includes residues within `distance_cutoff` Å of any ligand heavy atom.

    Args:
        protein_pdb (str): Path to the protein PDB file.
        ligand_sdf (str): Path to the ligand SDF file.
        distance_cutoff (float): Cutoff distance (Å) for defining the pocket.
        keep_hetatoms (bool): Whether to keep non-protein HETATM molecules.
        return_as_string (bool): If True, returns pocket PDB as a string instead of saving.
        output_dir (str | None): Directory to save pocket PDBs. Will not save if None.
        noise (float): Noise level for pocket extraction in Angstroms. Applied to the distance.
        filter_outlier_factor (float): Factor for filtering outliers based on distance from centroid.
        random_sample (bool): If True, randomly sample a pocket from the extracted pockets.
        If False, return all pockets.

    Returns:
        dict: Mapping of ligand indices to pocket PDB strings (if return_as_string=True) or file paths.
    """
    # TODO could add another cutoff to ensure Calpha is at least XA way -> Would remove other side-chain amino acids.
    if not return_as_string and output_dir is None:
        output_dir = os.path.dirname(protein_pdb)
        os.makedirs(output_dir, exist_ok=True)

    # For tracking, assumes <PDB_ID>_<LIG_ID>_<...>.pdb
    pdb_id = os.path.basename(protein_pdb).split("_")[0]
    lig_id = os.path.basename(protein_pdb).split("_")[1]
    # Load ligands
    ligand_mols: list[Chem.Mol] = read_ligands_from_sdf(ligand_sdf)
    if not ligand_mols:
        raise ValueError(f"No valid ligands found in SDF file: {ligand_sdf}")
    if random_sample:
        rand_id = np.random.randint(0, len(ligand_mols))
        ligand_mols = [ligand_mols[rand_id]]

    # Load protein
    structure = get_protein(protein_pdb)

    pockets = []
    # all_residue_coords = []
    for lig_idx, ligand_mol in enumerate(ligand_mols):
        ligand_coords = get_coordinates(ligand_mol, heavy_only=True)
        if ligand_coords.size == 0:
            continue  # Skip ligands with no valid coordinates

        pocket_residues = set()
        # residue_coords = []

        for chain in structure.get_chains():
            for residue in chain.get_residues():
                if not PDB.is_aa(residue, standard=True) and not keep_hetatoms:
                    continue  # Skip non-amino acid residues unless keeping HETATMs

                res_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
                if res_coords.size == 0:
                    continue

                # Compute distances to ligand atoms
                distances = np.linalg.norm(res_coords[:, None, :] - ligand_coords[None, :, :], axis=-1)
                if np.any(distances + np.random.normal(0, noise, distances.shape) < distance_cutoff):
                    pocket_residues.add((chain.id, residue.id))
                    # residue_coords.append(res_coords)

        if filter_outlier_factor > 0:
            valid_residues_bool = filter_outlier_residues(
                list(pocket_residues), structure, filter_outlier_factor, keep_dist=distance_cutoff
            )
            # Filter pocket residues based on the boolean results
            pocket_residues = [res_id for res_id, is_valid in zip(pocket_residues, valid_residues_bool) if is_valid]
            # residue_coords = [
            #     coords for coords, is_valid in zip(residue_coords, valid_residues_bool) if is_valid
            # ]

        # all_residue_coords.append(residue_coords)

        io = PDB.PDBIO()
        io.set_structure(structure)

        class PocketSelect(PDB.Select):
            def accept_residue(self, residue: PDB.Residue) -> bool:
                return (residue.get_parent().id, residue.id) in pocket_residues  # noqa

        if return_as_string:
            pdb_buffer = StringIO()
            io.save(pdb_buffer, select=PocketSelect())
            pockets.append(pdb_buffer.getvalue())
        if output_dir:
            output_pdb = os.path.join(output_dir, f"{pdb_id}_{lig_id}_pocket_{lig_idx}.pdb")
            io.save(output_pdb, select=PocketSelect())

    return pockets, ligand_mols
    # return pockets, ligand_mols, all_residue_coords


def extract_pockets_kdtree(  # noqa
    protein_pdb: Union[str, Path],
    ligand_sdf: Union[str, Path],
    distance_cutoff: float = 10.0,
    keep_hetatoms: bool = True,
    return_as_string: bool = True,
    output_dir: Union[str, Path, None] = None,
    noise: float = 0.5,
    filter_outlier_factor: float = -1,
    random_sample: bool = False,
) -> tuple[list[str], list[Chem.Mol]]:
    """
    Fast version: Extracts binding pockets using KDTree-based spatial search.

    Args:
        protein_pdb: Path to the protein PDB file.
        ligand_sdf: Path to the ligand SDF file.
        distance_cutoff: Å distance cutoff for defining pocket.
        keep_hetatoms: If True, includes non-standard residues like metal ions.
        return_as_string: If True, returns PDBs as strings; else saves to disk.
        output_dir: Directory to save outputs if saving to disk.
        noise: Optional noise added to cutoff distance (Å).
        filter_outlier_factor: Optional filtering of outlier residues (unused here).
        random_sample: If True, randomly select one ligand pose.

    Returns:
        Tuple of (list of pocket PDB strings or paths, list of ligand mols)
    """

    protein_pdb = Path(protein_pdb)
    ligand_sdf = Path(ligand_sdf)

    if not return_as_string and output_dir is None:
        output_dir = protein_pdb.parent
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    pdb_id = protein_pdb.stem.split("_")[0]
    lig_id = protein_pdb.stem.split("_")[1] if "_" in protein_pdb.stem else "LIG"

    ligand_mols = read_ligands_from_sdf(ligand_sdf)
    if not ligand_mols:
        raise ValueError(f"No valid ligands found in: {ligand_sdf}")

    if random_sample:
        ligand_mols = [np.random.choice(ligand_mols)]

    structure = get_protein(protein_pdb)

    pockets = []
    for lig_idx, ligand_mol in enumerate(ligand_mols):
        ligand_coords = get_coordinates(ligand_mol, heavy_only=True)
        if ligand_coords.size == 0:
            continue

        # Build KDTree for fast lookup
        kdtree = cKDTree(ligand_coords)

        pocket_residues = set()
        for chain in structure.get_chains():
            for residue in chain.get_residues():
                if not PDB.is_aa(residue, standard=True) and not keep_hetatoms:
                    continue

                atom_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
                if atom_coords.size == 0:
                    continue

                # Check if *any* atom in this residue is close to the ligand
                neighbors_found = any(
                    len(kdtree.query_ball_point(coord, r=distance_cutoff + noise)) > 0 for coord in atom_coords
                )
                if neighbors_found:
                    pocket_residues.add((chain.id, residue.id))

        if filter_outlier_factor > 0:
            valid_residues_bool = filter_outlier_residues(
                list(pocket_residues), structure, filter_outlier_factor, keep_dist=distance_cutoff
            )
            pocket_residues = [res_id for res_id, is_valid in zip(pocket_residues, valid_residues_bool) if is_valid]

        io = PDB.PDBIO()
        io.set_structure(structure)

        class PocketSelect(PDB.Select):
            def accept_residue(self, residue: PDB.Residue) -> bool:
                return (residue.get_parent().id, residue.id) in pocket_residues  # noqa: B023

        if return_as_string:
            pdb_buffer = StringIO()
            io.save(pdb_buffer, select=PocketSelect())
            pockets.append(pdb_buffer.getvalue())
        else:
            output_pdb = Path(output_dir) / f"{pdb_id}_{lig_id}_pocket_{lig_idx}.pdb"
            io.save(output_pdb, select=PocketSelect())
            pockets.append(str(output_pdb))

    return pockets, ligand_mols


def extract_pockets_vectorised(  # noqa
    protein_pdb: Union[str, Path],
    ligand_sdf: Union[str, Path],
    distance_cutoff: float = 10.0,
    keep_hetatoms: bool = True,
    return_as_string: bool = True,
    output_dir: Union[str, Path] | None = None,
    noise: float = 0.5,
    filter_outlier_factor: float = -1,  # unused placeholder
    random_sample: bool = False,
) -> tuple[Union[set[tuple[str, tuple]], str], list[Chem.Mol]]:
    """
    Extract binding pockets from a protein PDB using ligand coordinates with a KDTree-based approach.

    Args:
        protein_pdb (Union[str, Path]): Path to the protein PDB file.
        ligand_sdf (Union[str, Path]): Path to the ligand SDF file.
        distance_cutoff (float): Base distance cutoff (in Å) for including a protein atom.
        keep_hetatoms (bool): Whether to include non-standard residues (HETATMs).
        return_as_string (bool): If True, returns pocket PDB as a string (or writes to a file if output_dir is provided)
                                   If False, returns a set of (chain_id, residue.id) tuples.
        output_dir (Optional[Union[str, Path]]): If provided (and return_as_string is True),
            the pocket is saved to this directory.
        noise (float): Additional margin to add to the cutoff (in Å).
        filter_outlier_factor (float): Unused in this implementation.
        random_sample (bool): If True, randomly sample one ligand molecule from the SDF file.

    Returns:
        Tuple[Union[Set[Tuple[str, Tuple]], str], List[Chem.Mol]]:
            A tuple where the first element is either:
              - a set of (chain_id, residue.id) tuples representing pocket residues, if return_as_string is False, or
              - a PDB-formatted string (or a file path string if output_dir is provided) containing the pocket,
            and the second element is the list of ligand molecule objects.
    """
    protein_pdb = Path(protein_pdb)
    ligand_sdf = Path(ligand_sdf)

    if output_dir is not None and return_as_string:
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    ligand_mols = read_ligands_from_sdf(ligand_sdf)
    if not ligand_mols:
        raise ValueError(f"No ligands found in {ligand_sdf}")
    if random_sample:
        ligand_mols = [np.random.choice(ligand_mols)]

    structure = get_protein(protein_pdb)

    # Build protein KDTree once
    protein_coords = []
    atom_to_residue = []
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            if not PDB.is_aa(residue, standard=True) and not keep_hetatoms:
                continue
            for atom in residue.get_atoms():
                if atom.element == "H":
                    continue
                protein_coords.append(atom.get_coord())
                atom_to_residue.append((chain.id, residue.id))

    if not protein_coords:
        return [], ligand_mols

    protein_coords = np.array(protein_coords)
    protein_kdtree = cKDTree(protein_coords)

    # Reuse protein structure
    io = PDBIO()
    io.set_structure(structure)

    class PocketSelect(Select):
        def __init__(self, pocket_residues: set[tuple[str, tuple]]) -> None:
            super().__init__()
            self.pocket_residues = pocket_residues

        def accept_residue(self, residue):  # noqa
            return (residue.get_parent().id, residue.id) in self.pocket_residues

    pocket_strings = []
    for idx, mol in enumerate(ligand_mols):
        ligand_coords = get_coordinates(mol, heavy_only=True)
        if ligand_coords.size == 0:
            pocket_strings.append("")  # Empty pocket
            continue

        indices_list = protein_kdtree.query_ball_point(ligand_coords, r=distance_cutoff + noise)
        unique_indices = {i for sub in indices_list for i in sub}
        pocket_residues = {atom_to_residue[i] for i in unique_indices}

        if not return_as_string:
            raise NotImplementedError("Only string output supported in this version")

        if output_dir:
            out_file = output_dir / f"{protein_pdb.stem}_lig{idx}_pocket.pdb"
            io.save(str(out_file), select=PocketSelect(pocket_residues))
            pocket_strings.append(str(out_file))
        else:
            buffer = StringIO()
            io.save(buffer, select=PocketSelect(pocket_residues))
            pocket_strings.append(buffer.getvalue())

    return pocket_strings, ligand_mols


def extract_pocket_com(
    structure: PDB.Structure,
    ligand_coords: np.ndarray,
    distance_cutoff: float,
    heavy_atoms_only: bool = True,
    keep_hetatoms: bool = False,
) -> np.ndarray:
    """
    Calculate the center of mass (CoM) of the pocket residues based on a distance cutoff to the ligand.

    Args:
        structure (Bio.PDB.Structure): The protein structure to extract the pocket from.
        ligand_coords (np.ndarray): Coordinates of the heavy atoms of the bound ligand.
        distance_cutoff (float): Cutoff distance to define which residues belong to the pocket.
        heavy_atoms_only (bool): Whether to consider only heavy atoms.
        keep_hetatoms (bool): Whether to include heteroatoms (cofactors, ligands, etc.) in the pocket.

    Returns:
        np.ndarray: The center of mass (CoM) of the pocket.
    """
    pocket_coords = []

    # Iterate over all chains and residues in the first model
    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip residues that are not amino acids or, if not keeping hetatoms, skip heteroatoms
                if not PDB.is_aa(residue, standard=True) and (not keep_hetatoms or residue.get_id()[0] != " "):
                    continue  # Skip non-standard amino acids and heteroatoms (unless keep_hetatoms is True)

                # Now, go over the atoms in the residue
                for atom in residue.get_atoms():
                    element = atom.element.strip()  # Remove any extra spaces
                    if heavy_atoms_only and element.upper() == "H":
                        continue  # Skip hydrogen atoms if heavy_atoms_only is True
                    coord = atom.get_coord()  # Get atom coordinates

                    # Compute the distance to the ligand atoms and check if it's within the cutoff
                    dists = np.linalg.norm(ligand_coords - coord, axis=1)
                    if np.min(dists) < distance_cutoff:
                        pocket_coords.append(coord)  # Add the coordinate to the pocket list

    # If there are no coordinates in the pocket, return a zero vector
    if not pocket_coords:
        return np.zeros(3)

    # Convert lists to arrays for easier manipulation
    pocket_coords = np.array(pocket_coords, dtype=np.float32)
    com = pocket_coords.mean(axis=0)

    return com


def read_pdb_from_string(pdb_string: str, as_biopython: bool = False) -> Union[Chem.Mol, PDB.Structure]:
    """Parses a PDB string into a Biopython Structure | Mol object."""
    tmp = None
    if as_biopython:
        parser = PDB.PDBParser(QUIET=True)
        tmp = parser.get_structure("protein", StringIO(pdb_string))
    else:
        try:
            tmp = Chem.MolFromPDBBlock(pdb_string, removeHs=True)
            assert tmp is not None
        except Exception as e:
            # NOTE disabled non-sanitized training. This will destroy bond and atom features!
            return None
            print(f"Failed to parse PDB string: {e}. Skipping Sanitization")
            try:
                tmp = Chem.MolFromPDBBlock(pdb_string, removeHs=False, sanitize=False)
                assert tmp is not None
            except Exception as e:
                print(f"Failed to parse PDB string without sanitization: {e}.")
    return tmp


def read_and_clean_pdb_from_string(
    pdb_str: str,
    sanitize: bool = True,
    remove_hs: bool = True,
    remove_waters: bool = True,
    remove_metals: bool = True,
) -> Chem.Mol:
    """
    Convert a PDB string into an RDKit Mol object with options to remove water molecules and metal atoms.

    Args:
        pdb_str: Contents of a PDB file as a string.
        sanitize: Whether to sanitize the molecule.
        remove_hs: Whether to remove explicit hydrogens.
        remove_waters: Remove HOH/WAT residues.
        remove_metals: Remove metal atoms by element symbol.

    Returns:
        RDKit Mol object or None if parsing fails.
    """
    mol = Chem.MolFromPDBBlock(pdb_str, sanitize=False, removeHs=False)
    if mol is None:
        return None

    editable = Chem.EditableMol(mol)
    atoms_to_remove = []

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        res_info = atom.GetPDBResidueInfo()
        res_name = res_info.GetResidueName().strip() if res_info else ""

        if remove_waters and res_name in {"HOH", "WAT"}:  # noqa
            atoms_to_remove.append(atom.GetIdx())
        elif remove_metals and symbol in {"Zn", "Mg", "Fe", "Cu", "Mn", "Ca", "Na", "K", "Co", "Ni"}:
            atoms_to_remove.append(atom.GetIdx())

    for idx in sorted(atoms_to_remove, reverse=True):
        editable.RemoveAtom(idx)

    mol = editable.GetMol()
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"Protein Sanitization failed: {e}")
            return None

    if remove_hs:
        mol = Chem.RemoveHs(mol)

    return mol


def find_nearby_hetatoms(  # noqa: C901
    protein_pdb: str | Path,
    ligand_sdf: str | Path,
    distance_cutoff: float = 6.0,
    noise: float = 0.0,
    random_sample: bool = False,
    debug: bool = False,
) -> tuple[list[Chem.Mol], list[list[tuple[int, str, tuple, str, float]]]]:
    """
    Find hetero residues (HETATM groups) that are within distance_cutoff
    of any ligand heavy atom.

    Returns:
      (ligand_mols, hets_per_ligand)

      - ligand_mols: list[rdkit.Chem.Mol]
      - hets_per_ligand: list (len = #ligands) of lists of tuples:
           (model_id:int, chain_id:str, residue_id:tuple, resname:str, min_distance:float)

    Notes:
      - residue_id is the Bio.PDB residue.id tuple, typically ('H_X', resseq, icode).
      - min_distance is the minimum atom-to-atom distance between that het residue and the ligand.
      - Uses Bio.PDB.PDBParser to parse the raw PDB (preserves HETATM).
    """

    # read ligands
    ligand_mols: list[Chem.Mol] = read_ligands_from_sdf(ligand_sdf)
    if not ligand_mols:
        raise ValueError(f"No valid ligands found in SDF: {ligand_sdf}")
    if random_sample:
        rand_id = np.random.randint(0, len(ligand_mols))
        ligand_mols = [ligand_mols[rand_id]]

    # parse PDB directly with Biopython so HETATM rows are preserved
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(Path(protein_pdb).stem, str(protein_pdb))

    hets_per_ligand: list[list[tuple[int, str, tuple, str, float]]] = []

    for lig_idx, lig in enumerate(ligand_mols):
        lig_coords = get_coordinates(lig, heavy_only=True)
        if lig_coords is None or lig_coords.size == 0:
            hets_per_ligand.append([])
            if debug:
                print(f"[debug] ligand {lig_idx} has no coords, skipping")
            continue

        found = []
        # iterate models/chains/residues — robust across TER blocks
        for model in structure:
            for chain in model:
                for residue in chain:
                    hetflag = residue.id[0]
                    # only consider hetero residues: hetflag != ' ' (space) indicates hetero
                    if hetflag == " ":
                        continue

                    atoms = list(residue.get_atoms())
                    if len(atoms) == 0:
                        continue

                    res_coords = np.array([a.get_coord() for a in atoms])  # (n_res_atoms, 3)
                    # pairwise distances: (n_res_atoms, n_lig_atoms)
                    dists = np.linalg.norm(res_coords[:, None, :] - lig_coords[None, :, :], axis=-1)
                    if noise and noise > 0:
                        dists = dists + np.random.normal(0, noise, dists.shape)

                    min_dist = float(np.min(dists))
                    if min_dist <= distance_cutoff:
                        # collect: (model_id, chain_id, residue.id, resname, min_distance)
                        found.append((model.id, chain.id, residue.id, residue.get_resname(), min_dist))

        # sort by increasing distance (closest first)
        found.sort(key=lambda x: x[-1])
        hets_per_ligand.append(found)

        if debug:
            print(f"[debug] ligand {lig_idx}: found {len(found)} hetero residues within {distance_cutoff} Å")
            for info in found[:30]:
                print("  ", info)

    return ligand_mols, hets_per_ligand


# -----------------------------------------------------
# ---------------------- COMPLEX ----------------------
# -----------------------------------------------------


def sample_complex(
    *,
    sdf: str | Path,
    pdb: str | Path,
    keep_hetatoms: bool = True,
    distance_cutoff: float = 10.0,
    distance_noise: float = 0.5,
    filter_outlier_factor: float = -1,
) -> tuple[str, Chem.Mol]:
    """Sample complex from sdf & pdb pair."""

    # NOTE this is a deprecated function. Use extract_pockets_kdtree instead.
    # Load pocket if pdb has pocket, else create pocket from sdf
    # if "pocket" in str(pdb):
    #     pocket = get_protein(pdb)
    #     ligands = read_ligands_from_sdf(sdf)
    #     assert len(ligands) == 1
    #     ligand = ligands[0]
    # else:

    # pockets, ligands = extract_pockets_vectorised(
    # pockets, ligands = extract_pockets_kdtree(
    pockets, ligands = extract_pockets_deprecated(
        pdb,
        sdf,
        distance_cutoff=distance_cutoff,
        keep_hetatoms=keep_hetatoms,
        return_as_string=True,
        output_dir=None,
        noise=distance_noise,
        filter_outlier_factor=filter_outlier_factor,
        random_sample=True,
    )
    print(f"pocket: {pockets}")

    assert len(pockets) > 0, f"No pockets found in {pdb}."
    assert len(ligands) > 0, f"No ligands found in {sdf}."
    assert len(pockets) == len(ligands), f"Mismatch between pockets and ligands in {pdb}."

    # # Sample from the pockets if more than one
    # idx = np.random.randint(0, len(pockets))
    # pocket, ligand = pockets[idx], ligands[idx]
    pocket = pockets[0]
    ligand = ligands[0]

    return pocket, ligand
