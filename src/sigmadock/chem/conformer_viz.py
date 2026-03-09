"""Conformer alignment and 3D viewing helpers for use in notebooks and scripts."""
from __future__ import annotations

from typing import Optional

import numpy as np
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem


def view_molecule(mol: Chem.Mol, style: str = "stick", width: int = 800, height: int = 400) -> py3Dmol.view:
    """Visualize a molecule using py3Dmol."""
    view = py3Dmol.view(width=width, height=height)
    view.addModel(Chem.MolToMolBlock(mol), "mol")
    view.setStyle({style: {}})
    view.zoomTo()
    return view


def view_conformations(
    mol: Chem.Mol, style: str = "stick", width: int = 800, height: int = 400
) -> py3Dmol.view:
    """Visualize a molecule with multiple conformations as separate models."""
    view = py3Dmol.view(width=width, height=height)
    for conf in mol.GetConformers():
        view.addModel(Chem.MolToMolBlock(mol, confId=conf.GetId()), "mol")
    view.setStyle({style: {}})
    view.zoomTo()
    return view


def clone_mol_with_conformers(mol: Chem.Mol) -> Chem.Mol:
    """Return a clone of mol preserving conformers (so we can modify coords in place)."""
    clone = Chem.Mol(mol)
    clone.RemoveAllConformers()
    for conf in mol.GetConformers():
        clone.AddConformer(Chem.Conformer(conf), assignId=True)
    return clone


def _get_coords_array(mol: Chem.Mol, conf_id: int, atom_indices: list[int]) -> np.ndarray:
    conf = mol.GetConformer(int(conf_id))
    return np.array([conf.GetAtomPosition(i) for i in atom_indices], dtype=float)


def _apply_rigid_transform_to_conf(mol: Chem.Mol, conf_id: int, R: np.ndarray, t: np.ndarray) -> None:
    conf = mol.GetConformer(int(conf_id))
    for i in range(conf.GetNumAtoms()):
        p = np.array(conf.GetAtomPosition(i), dtype=float)
        conf.SetAtomPosition(i, tuple(R.dot(p) + t))


def _kabsch_align(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find rotation R and translation t that maps points A -> B in least-squares sense."""
    assert A.shape == B.shape
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    A_c = A - centroid_A
    B_c = B - centroid_B
    H = A_c.T @ B_c
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t


def align_conformers(
    mol: Chem.Mol,
    ref_conf_id: int = 0,
    atom_indices: Optional[list[int]] = None,
    align_heavy_atoms: bool = True,
    clone_molecule: bool = True,
) -> tuple[Chem.Mol, np.ndarray]:
    """
    Align conformers to ref_conf_id. Returns (aligned_mol, rmsds_array).
    If clone_molecule=True, returns a clone (original unchanged).
    """
    mol_aligned = clone_mol_with_conformers(mol) if clone_molecule else mol
    if atom_indices is None:
        atom_indices = [
            a.GetIdx() for a in mol_aligned.GetAtoms()
            if (a.GetAtomicNum() > 1) or not align_heavy_atoms
        ]
    if len(atom_indices) == 0:
        atom_indices = [a.GetIdx() for a in mol_aligned.GetAtoms()]

    conf_ids = [c.GetId() for c in mol_aligned.GetConformers()]
    try:
        AllChem.AlignMolConformers(mol_aligned, refId=int(ref_conf_id), atomIds=atom_indices)
        ref_coords = _get_coords_array(mol_aligned, ref_conf_id, atom_indices)
        rmsds = []
        for cid in conf_ids:
            coords = _get_coords_array(mol_aligned, cid, atom_indices)
            rmsd = np.sqrt(np.mean(np.sum((coords - ref_coords) ** 2, axis=1)))
            rmsds.append(rmsd)
        return mol_aligned, np.array(rmsds)
    except Exception:
        ref_coords = _get_coords_array(mol_aligned, ref_conf_id, atom_indices)
        rmsds = []
        for cid in conf_ids:
            if int(cid) == int(ref_conf_id):
                rmsds.append(0.0)
                continue
            coords = _get_coords_array(mol_aligned, cid, atom_indices)
            R, t = _kabsch_align(coords, ref_coords)
            _apply_rigid_transform_to_conf(mol_aligned, cid, R, t)
            coords_aligned = _get_coords_array(mol_aligned, cid, atom_indices)
            rmsd = np.sqrt(np.mean(np.sum((coords_aligned - ref_coords) ** 2, axis=1)))
            rmsds.append(rmsd)
        return mol_aligned, np.array(rmsds)


def view_aligned_conformers(
    mol: Chem.Mol,
    conf_ids: Optional[list[int]] = None,
    ref_conf_id: int = 0,
    atom_indices: Optional[list[int]] = None,
    style: str = "stick",
    width: int = 900,
    height: int = 400,
    color_by_model: bool = True,
    clone_and_align: bool = True,
) -> py3Dmol.view:
    """Align conformers and return a py3Dmol view with each conformer as a separate model."""
    all_conf_ids = [c.GetId() for c in mol.GetConformers()]
    conf_ids = all_conf_ids if conf_ids is None else [int(x) for x in conf_ids]
    mol_aligned, rmsds = align_conformers(
        mol, ref_conf_id=ref_conf_id, atom_indices=atom_indices, clone_molecule=clone_and_align
    )
    view = py3Dmol.view(width=width, height=height)
    for i, cid in enumerate(conf_ids):
        view.addModel(Chem.MolToMolBlock(mol_aligned, confId=int(cid)), "sdf")
    for i in range(len(conf_ids)):
        if color_by_model:
            view.setStyle({"model": i}, {style: {"colorscheme": "Jmol"}})
        else:
            view.setStyle({"model": i}, {style: {}})
    view.zoomTo()
    view._rmsds = rmsds  # noqa: SLF001
    return view
