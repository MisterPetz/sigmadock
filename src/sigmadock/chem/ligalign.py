"""Alignment of ligand based on torsional update of rotatable bonds
This module provides a class for optimizing the conformation of a ligand.
Useful for determining the optimal conformation of a ligand in a binding pocket.
"""

from __future__ import annotations

import copy
import logging

import networkx as nx
import numpy as np
from PIL import Image
from posebusters import PoseBusters
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, rdMolTransforms
from rdkit.Geometry import Point3D
from scipy.optimize import differential_evolution

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO extend for better non-planar ring substitution. Chair & Boat.


def get_non_planar_rings(mol: Chem.Mol) -> list[list[int]]:
    """
    Identify rings that are likely to adopt multiple conformations due to high symmetry or flexibility.

    Criteria:
    - Rings with low unique atom rankings (indicating symmetry)
    - Rings with flexible torsions (non-aromatic, spiro, or fused systems)
    """

    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()

    if not rings:
        return []

    atom_ranks = list(Chem.CanonicalRankAtoms(mol))
    symmetric_rings = []

    for ring in rings:
        ranks = [atom_ranks[idx] for idx in ring]
        unique_ranks = len(set(ranks))

        # Condition: If the ring has significantly repeated ranks, it is symmetric
        if unique_ranks <= len(ring) // 2:
            symmetric_rings.append(ring)
            continue  # No need to check further if it's already symmetric

        # Check if the ring is non-aromatic (flexible)
        if not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            symmetric_rings.append(ring)
            continue

        # Check if the ring is part of a fused system (increases flexibility)
        neighboring_rings = [r for r in rings if set(r) & set(ring) and r != ring]
        if neighboring_rings:
            symmetric_rings.append(ring)

    return symmetric_rings


class ConformerOptimizer:
    def __init__(
        self,
        mol: Chem.Mol,
        tries: int = 10,
        popsize: int = 15,
        maxiter: int = 50,
        tolerance: float = 0.25,
        max_energy_delta: float = 5.0,
        seed: int | None = None,
        remove_hs: bool = True,
        ring_matching: bool = False,
        ignore_conjugated: bool = False,
        pb_check: PoseBusters | None = None,
    ) -> None:
        self.ignore_conjugated: bool = ignore_conjugated
        self.orig_mol: Chem.Mol = copy.deepcopy(mol)
        self.ref_mol: Chem.Mol = self._prepare_reference(mol, remove_hs)
        self.rotatable_bonds: list[tuple[int, int, int, int]] = self._find_rotatable_bonds(self.ref_mol)
        self.params: dict = {
            "tries": tries,
            "popsize": popsize,
            "maxiter": maxiter,
            "tolerance": tolerance,
            "max_energy_delta": max_energy_delta,
            "remove_hs": remove_hs,
            "seed": seed,
            "ring_matching": ring_matching,
        }
        self.pb_check: PoseBusters | None = pb_check

    def _prepare_reference(self, mol: Chem.Mol, remove_hs: bool) -> Chem.Mol:
        ref = Chem.RemoveHs(mol, sanitize=True) if remove_hs else copy.deepcopy(mol)
        # return AllChem.RemoveAllHs(ref)
        # NOTE AllChem.RemoveAllHs(ref) strips stereochemical hydrogens!!
        return ref

    def _find_rotatable_bonds(self, mol: Chem.Mol) -> list[tuple[int, int, int, int]]:  # noqa: C901
        # Ensure proper bond conjugation perception
        Chem.GetSSSR(mol)
        mol.UpdatePropertyCache()

        # Build molecular graph
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

        rot_bonds = []
        for e in G.edges():
            # Get corresponding RDKit bond object
            bond = mol.GetBondBetweenAtoms(*e)
            if not bond:
                continue

            # Skip conjugated bonds and non-single bonds
            if bond.GetIsConjugated() and self.ignore_conjugated:
                a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if not mol.GetAtomWithIdx(a1).IsInRing() and not mol.GetAtomWithIdx(a2).IsInRing():
                    # Only skip conjugated bonds that are not in / stemming from rings (weak conjugation)
                    continue

            # Skip non-single bonds
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            # Check if bond is in a ring
            if bond.IsInRing():
                continue

            # Perform graph connectivity check
            G2 = copy.deepcopy(G)
            G2.remove_edge(*e)
            if nx.is_connected(G2):
                continue

            # Check component sizes
            components = list(nx.connected_components(G2))
            if any(len(c) < 2 for c in components):
                continue

            # Get neighboring atoms
            try:
                n0 = next(G2.neighbors(e[0]))
                n1 = next(G2.neighbors(e[1]))
                rot_bonds.append((n0, e[0], e[1], n1))
            except StopIteration:
                continue

        return rot_bonds

    def optimize_torsions(
        self,
        pocket: Chem.Mol | None = None,
        strict: bool = False,
    ) -> tuple[Chem.Mol, float, float, bool]:
        mols: list[Chem.Mol] = []
        rmsds: list[float] = []
        energies: list[float] = []
        pb_checks: list[bool] = []

        # If no tries are specified, return the original molecule and 0.0 RMSD
        if not self.params["tries"]:
            return self.ref_mol, 0.0, 0.0

        ref_energy = self.compute_energy(self.ref_mol)
        # Start with experimentally preferred coordinates
        rand_coords = False
        for idx in range(self.params["tries"]):
            if idx > 0:
                rand_coords = True
            probe = (
                self._generate_template_conformer()
                if self.params["ring_matching"]
                else self._generate_conformer(random_coords=rand_coords, idx=idx)
            )
            optimized = self._optimize_conformer(probe)
            aligned_conformer, rmsd = self._align_and_calculate_rmsd(optimized)
            # Replace conformers for aligned conformer
            optimized.RemoveAllConformers()
            optimized.AddConformer(aligned_conformer, assignId=True)
            # PB Check
            if self.pb_check is None:
                pb_valid = True
            else:
                pb_valid: bool = (
                    self.pb_check.bust(
                        mol_true=self.ref_mol,
                        mol_pred=optimized,
                        mol_cond=pocket,
                    )
                    .transpose()
                    .mean()
                    .values[0]
                    == 1
                )

            energy_delta = self.compute_energy(optimized) - ref_energy
            mols.append(optimized)
            rmsds.append(rmsd)
            energies.append(energy_delta)
            pb_checks.append(pb_valid)
            if (rmsd < self.params["tolerance"]) and pb_valid:
                if strict:
                    # If strict, we want to find the best conformer unless we have a valid one
                    if energy_delta < self.params["max_energy_delta"]:
                        # If we have a valid conformer, we can break early
                        break
                else:
                    # If not strict, we can break early
                    break

        # TODO Softmax probabilistic selection for indices with RMSD < threshold for better diversity.
        best_idx = np.argmin(rmsds)
        best_mol, best_rmsd, best_energy, pb_val = (
            mols[best_idx],
            rmsds[best_idx],
            energies[best_idx],
            pb_checks[best_idx],
        )
        if strict:
            if (best_rmsd < self.params["tolerance"]) and (best_energy < self.params["max_energy_delta"]) and pb_val:
                return best_mol, best_rmsd, best_energy, pb_val
            else:
                print(
                    f"No suitable conformation found within the specified tolerance: \
                        R = {self.params['tolerance']}, E = {self.params['max_energy_delta']}. PB-Val = {pb_val} \
                        Returning original molecule."
                )
                return self.ref_mol, 0.0, 0.0, True
        else:
            return (
                best_mol,
                best_rmsd,
                best_energy,
                pb_val,
            )

    def compute_energy(self, mol: Chem.Mol) -> float:
        """
        Compute the energy of the molecule using MMFF force field.

        Args:
            mol: The molecule for which to compute the energy.

        Returns:
            energy: The computed energy.
        """
        # Create a deep copy of the molecule to preserve the original.
        mol = copy.deepcopy(mol)
        mol = Chem.AddHs(mol, addCoords=True)

        # Set up the MMFF properties and force field.
        # ff = None
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
        if ff is None:
            if mmff_props is None:
                logger.log(0, "[WARN] MMFFGetMoleculeForceField returned None despite valid parameters.")
            else:
                logger.log(0, "[WARN] MMFFGetMoleculeForceField returned None. Attempting UFF.")

        # If MMFF force field is not available, fall back to UFF.
        if ff is None:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff is None:
                raise ValueError(
                    "Could not set up either MMFF or UFF force fields. "
                    "Check that your molecule has a valid 3D conformer and supported atom types."
                )

        # Compute and return the energy.
        return ff.CalcEnergy()

    def optimize_conformation(  # noqa: C901
        self, query: Chem.Mol | None = None, max_iters: int = 100
    ) -> tuple[Chem.Mol, float, float]:
        """
        Optimize the bound ligand (reference molecule) using the MMFF force field.

        Returns:
            optimized_mol: The optimized molecule.
            rmsd: The RMSD between the original and optimized conformers.
            energy_change: The difference between the pre- and post-minimization energies.
        """
        if query is None:
            query = self.ref_mol

        # Create a deep copy to preserve the original molecule.
        mol_to_optimize = copy.deepcopy(query)
        # Add hydrogens
        mol_to_optimize = Chem.AddHs(mol_to_optimize, addCoords=True)

        energy_before = float("inf")
        energy_after = float("inf")
        ff_success = False

        # --- Step 1: Try to set up and run the more accurate MMFF force field ---
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol_to_optimize)
            ff = AllChem.MMFFGetMoleculeForceField(mol_to_optimize, mmff_props)
            if ff is not None:
                energy_before = ff.CalcEnergy()
                status = ff.Minimize(maxIters=max_iters)  # Using Minimize() with more iterations
                energy_after = ff.CalcEnergy()
                ff_success = status == 0  # Success if minimization converged
                if not ff_success:
                    print(f"[WARN] MMFF optimization did not converge after {max_iters} iterations (status: {status}).")
        except Exception:
            # This will catch errors from MMFFGetMoleculeProperties or CalcEnergy
            ff_success = False

        # --- Step 2: Fallback to UFF if MMFF failed ---
        if not ff_success:
            try:
                # print("[INFO] Falling back to UFF for optimization.")
                ff = AllChem.UFFGetMoleculeForceField(mol_to_optimize)
                if ff is None:
                    raise ValueError("UFF setup failed.")
                # Use the last calculated energy_before if available, otherwise calculate new
                if energy_before == float("inf"):
                    energy_before = ff.CalcEnergy()
                status = ff.Minimize(maxIters=max_iters)
                energy_after = ff.CalcEnergy()
                if status != 0:
                    print(f"[WARN] UFF optimization did not converge after {max_iters} iterations (status: {status}).")

            except Exception as e:
                print(f"[ERROR] Both MMFF and UFF optimization failed. Final error: {e}")
                return Chem.RemoveHs(query), 0.0, 0.0

        # Calculate RMSD against the original query molecule
        try:
            # Aligning before calculating RMSD is crucial
            rmsd = AllChem.AlignMol(Chem.RemoveHs(mol_to_optimize), Chem.RemoveHs(query))
        except Exception as e:
            print(f"[ERROR] Could not align molecules for RMSD calculation: {e}")
            rmsd = float("inf")

        return Chem.RemoveHs(mol_to_optimize), rmsd, energy_before - energy_after

    def _apply_bound_ring_coords(self, mol: Chem.Mol, ref_mol: Chem.Mol) -> Chem.Mol:
        """
        For each ring in the molecule, copy the coordinates from the corresponding atoms
        in the bound (reference) ligand. (Assumes the atom ordering is identical.)
        """
        conf = mol.GetConformer()
        ref_conf = ref_mol.GetConformer()
        ring_info = mol.GetRingInfo()
        # For each ring, update all atom positions from the reference
        for ring in ring_info.AtomRings():
            for idx in ring:
                ref_pos = ref_conf.GetAtomPosition(idx)
                conf.SetAtomPosition(idx, Point3D(ref_pos.x, ref_pos.y, ref_pos.z))
        return mol

    def _generate_conformer(self, random_coords: bool = False, idx: int | None = None) -> Chem.Mol:
        mol = copy.deepcopy(self.orig_mol)
        mol.RemoveAllConformers()
        mol = AllChem.AddHs(mol)

        ps = AllChem.ETKDGv3()
        if getattr(ps, "useRandomCoords", None) is not None:
            # Use the random coordinates option if available
            ps.useRandomCoords = random_coords
        elif getattr(ps, "UseRandomCoords", None) is None:
            # Older versions of RDKit may not have this attribute instead
            ps.UseRandomCoords = random_coords
        else:
            raise ValueError("Random coordinates option not available in this version of RDKit.")
        if self.params["seed"] is not None:
            ps.randomSeed = self.params["seed"]
            if idx is not None:
                ps.randomSeed += idx
        else:
            ps.randomSeed = idx if idx is not None else 0

        failures = 0
        while failures < 3:
            if AllChem.EmbedMolecule(mol, ps) != -1:
                break
            failures += 1
            ps.useRandomCoords = True
        else:
            AllChem.EmbedMolecule(mol, ps)
            AllChem.MMFFOptimizeMolecule(mol, confId=0)

        # Minimize conformer -> Helps for structure matching
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
        mol = Chem.RemoveHs(mol, sanitize=True)
        # mol = AllChem.RemoveAllHs(mol)
        # NOTE AllChem.RemoveAllHs(mol) strips stereochemical hydrogens!!
        return mol

    def _generate_template_conformer(self) -> Chem.Mol:
        mol = copy.deepcopy(self.orig_mol)
        mol = AllChem.AddHs(mol)

        # Identify rings that need relative constraints
        constrained_rings = get_non_planar_rings(self.ref_mol)
        if not constrained_rings:
            return self._generate_conformer()  # No constraints needed

        # Flatten ring indices
        core_indices = sorted({idx for ring in constrained_rings for idx in ring})
        # Extract the correct core with correct atom ordering
        core = Chem.PathToSubmol(self.ref_mol, core_indices)

        # Get atom mapping (ensures correct correspondence between `mol` and `core`)
        match = mol.GetSubstructMatch(core)
        if not match:
            raise ValueError("Core mismatch! Could not align substructure.")

        try:
            # Optionally, use ETKDG parameters with random coords if needed:
            constrained_mol = AllChem.ConstrainedEmbed(mol, core, useTethers=True, maxAttempts=10)

            # Check if embedding was successful by ensuring we have a conformer.
            if constrained_mol is None or constrained_mol.GetNumConformers() == 0:
                raise ValueError("Constrained embedding failed.")

            mol = constrained_mol

        except Exception as e:
            print("Constrained embedding failed:", e)
            print("Falling back to reference pose constraints with minimization.")
            # Fallback: try an unconstrained embedding and then MMFF minimization
            status = AllChem.EmbedMolecule(mol, useRandomCoords=True)
            if status != 0:
                print("Unconstrained embedding returned non-zero status:", status)
            AllChem.MMFFOptimizeMolecule(mol, confId=0)

        return Chem.RemoveHs(mol, sanitize=True)

    def _optimize_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        if not self.rotatable_bonds:
            return mol

        temp_mol = copy.deepcopy(mol)
        assert temp_mol.GetNumConformers() == 1, "Conformer not found in the molecule."
        # NOTE duplicate conformer to avoid modifying the original
        # temp_mol.AddConformer(mol.GetConformer(0))
        true_mol = copy.deepcopy(self.ref_mol)

        # - Note; Adapted from DiffDock.
        result = differential_evolution(
            lambda x: self._score_conformation(temp_mol, true_mol, x),
            bounds=[(-np.pi, np.pi)] * len(self.rotatable_bonds),
            maxiter=self.params["maxiter"],
            popsize=self.params["popsize"],
            mutation=(0.5, 1),
            recombination=0.8,
            disp=False,
        )

        return self._apply_changes(temp_mol, result.x)

    def _score_conformation(self, mol: Chem.Mol, true_mol: Chem.Mol, angles: np.ndarray) -> float:
        conf = mol.GetConformer(0)
        for i, bond in enumerate(self.rotatable_bonds):
            rdMolTransforms.SetDihedralRad(conf, *bond, angles[i])
        return AllChem.AlignMol(mol, true_mol)

    def _apply_changes(self, mol: Chem.Mol, angles: np.ndarray) -> Chem.Mol:
        new_mol = copy.deepcopy(mol)
        conf = new_mol.GetConformer(0)
        for i, bond in enumerate(self.rotatable_bonds):
            rdMolTransforms.SetDihedralRad(conf, *bond, angles[i])
        return new_mol

    def _align_and_calculate_rmsd(self, mol: Chem.Mol) -> tuple[Chem.rdchem.Conformer, float]:
        true_mol = copy.deepcopy(self.ref_mol)
        newID = true_mol.AddConformer(mol.GetConformer(0), assignId=True)
        rms_list: list[float] = []
        # Note this aligns
        AllChem.AlignMolConformers(true_mol, RMSlist=rms_list)
        return true_mol.GetConformer(newID), rms_list[0]


class TorsionalOptimizer(ConformerOptimizer):
    def __init__(
        self,
        *args,  # noqa
        **kwargs,  # noqa
    ) -> None:
        super().__init__(*args, **kwargs)

    def optimize(self) -> tuple[Chem.Mol, float]:
        best_mol: Chem.Mol | None = None
        best_rmsd: float = float("inf")

        mols: list[Chem.Mol] = []
        rmsds: list[float] = []

        for idx in range(self.params["tries"]):  # noqa
            probe = self._generate_conformer()
            optimized = self._optimize_conformer(probe)
            rmsd = self._calculate_rmsd(optimized)
            mols.append(optimized)
            rmsds.append(rmsd)
            if rmsd < self.params["tolerance"]:
                break

        best_idx = int(np.argmin(rmsds))
        if rmsds[best_idx] < best_rmsd:
            best_mol, best_rmsd = mols[best_idx], rmsds[best_idx]

        return best_mol, best_rmsd

    def _optimize_conformer(self, mol: Chem.Mol) -> Chem.Mol:
        if not self.rotatable_bonds:
            return mol

        temp_mol = copy.deepcopy(mol)
        temp_mol.AddConformer(mol.GetConformer(0))

        max_iter = self.params.get("max_coord_iter", 10)
        _ = self.params.get("grid_step", np.deg2rad(10))  # in radians
        tol = self.params.get("coord_tol", 1e-3)  # convergence tolerance in radians

        # Coordinate Descent: update each torsion in a loop
        for iteration in range(max_iter):  # noqa
            improvement = 0.0
            for bond in self.rotatable_bonds:
                conf = temp_mol.GetConformer(0)
                current_angle = rdMolTransforms.GetDihedralRad(conf, *bond)
                best_angle = current_angle
                best_score = self._score_conformation(temp_mol)

                # Perform grid search around current_angle within ±45° (π/4 radians)
                search_range = np.linspace(current_angle - np.pi / 4, current_angle + np.pi / 4, 9)
                for angle in search_range:
                    rdMolTransforms.SetDihedralRad(conf, *bond, angle)
                    score = self._score_conformation(temp_mol)
                    if score < best_score:
                        best_score = score
                        best_angle = angle
                rdMolTransforms.SetDihedralRad(conf, *bond, best_angle)
                improvement += abs(best_angle - current_angle)

            # If the total change is below tolerance, we assume convergence.
            if improvement < tol:
                break

        return temp_mol


def visualize_comparison(ref_mol: Chem.Mol, optimized_mol: Chem.Mol, size: tuple[int, int] = (300, 300)) -> Image.Image:
    ref_img = Draw.MolToImage(ref_mol, size=size, legend="Bound Pose")
    opt_img = Draw.MolToImage(optimized_mol, size=size, legend="Optimized Pose")
    composite = Image.new("RGB", (size[0] * 2, size[1]))
    composite.paste(ref_img, (0, 0))
    composite.paste(opt_img, (size[0], 0))
    return composite
