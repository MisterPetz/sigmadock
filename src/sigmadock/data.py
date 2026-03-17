# import copy
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from Bio import PDB
from posebusters import PoseBusters
from rdkit import Chem
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from sigmadock.chem.fragmentation import (
    detect_torsional_bonds,
    fragment_molecule,
    get_all_virtual_nodes_per_fragment,
    get_fragment_map,
    get_fragmented_anchors_dummies,
    get_fragments_as_mols,
    get_non_torsional_neighbors,
    get_torsional_neighbors,
    get_triangle_equality_mapping,
)
from sigmadock.chem.ligalign import ConformerOptimizer
from sigmadock.chem.parsing import (
    extract_pocket_com,
    get_coordinates,
    inspect_structure,
    read_ligands_from_sdf,
    read_pdb_from_string,
    sample_complex,
    structure_to_rdkit,
)
from sigmadock.chem.processing import get_global_interaction_graph, get_global_ligand_graph, get_global_protein_graph
from sigmadock.core.loaders import CustomDataLoader
from sigmadock.datafronts import DataFront
from sigmadock.oracle import MAX_TORSIONAL_BONDS, MAX_WEIGHT
from sigmadock.torch_utils import (
    CachedRecycleWrapper,
    IterableCachedRecycleWrapper,  # noqa (Deprecated)
    IterableDeterministicRecycleWrapper,  # noqa (Deprecated)
    tensorise_idxs,
)


def worker_init_fn(worker_id: int) -> None:
    # torch.initial_seed() is large; reduce to 32-bit for numpy
    seed = torch.initial_seed() % (2**32 - 1)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


class SigmaDataset(Dataset):
    def __init__(
        self,
        # Dataset
        datafront: DataFront,
        # Pocket Definition
        pocket_com_cutoff: float = 6.0,
        pocket_distance_cutoff: float = 6.0,
        pocket_com_noise: float = 0.5,
        pocket_distance_noise: float = 0.5,
        prot_coordinate_distance_noise: float = 0.05,
        lig_coordinate_distance_noise: float = 0,
        # Cutoffs / Outliers. If negative, they are ignored.
        pocket_residue_outlier_factor: float = -1.0,
        pocket_virtual_cutoff: float = 15.0,
        # Parsing
        fragmentation_strategy: Literal["random", "largest", "max", "smallest", "canonical"] = "random",
        streamloading: bool = True,
        keep_hetatoms: bool = False,
        use_esm_embeddings: bool = False,
        esm_embeddings_clip_range: Optional[tuple[float, float]] = None,
        esm_embeddings_scaling_factor: float = 1.0,
        # Triangulation
        ignore_triangulation: bool = False,
        triangulate_dummies: bool = False,
        # Bound Ligand Alignement
        alignment_tries: int = 3,
        alignment_rmsd_tolerance: float | None = 1.0,
        alignment_energy_tolerance: float | None = 10.0,
        force_alignment: bool = False,
        ignore_conjugated_torsion: bool = False,
        pb_check: bool = True,
        # Misc
        seed: int = 42,
        verbose: bool = False,
        force_retry: bool = False,
        random_rotation: bool = False,
        skip_bounds_check: bool = False,
        mirror_prob: float = 0,  # TODO: potentially might be be useful even though it breaks chemistry! 
        get_mol_info: bool = False,
        # Modality
        sample_conformer: bool = False,  # If True, will sample a random conformer for the ligand instead of using the bound pose. # noqa
        # Follow-on Kwargs,
        **ignored_kwargs: dict,
    ) -> None:
        """SigmaDock Dataset for Protein-Ligand Complexes.

        Args:
            datafront (DataFront): DataFront object containing the dataset.
            pocket_com_cutoff (float): Distance cutoff (PL) for center of mass of system.
            pocket_com_noise (float): Perturbation added to the center of mass of the pocket.
            pocket_distance_cutoff (float): Distance cutoff for pocket extraction.
            pocket_distance_noise (float): Noise added to the distance cutoff.
            pocket_residue_outlier_factor (float): Factor for filtering outliers in pocket residues.
            pocket_virtual_cutoff (float): Distance cutoff for virtual nodes in the pocket.
            coordinate_distance_noise (float): Noise added to the coordinates of the pocket.
            streamloading (bool): Whether to use streaming loading.
            keep_hetatoms (bool): Whether to keep heteroatoms in the pocket.
            fragmentation_strategy (str): Fragmentation strategy for ligand.
                Options:
                    "random", "largest", "smallest", "canonical".
            use_embeddings (bool): Whether to load ESM embeddings.
            ignore_conjugated_torsion (bool): Whether to ignore conjugated torsion in ligand fragmentation.
            alignment_tries (int): Number of tries for RDKit alignment. If set to 0, it will be disabled.
            alignment_rmsd_tolerance (float): Tolerance for RDKit alignment.
            force_alignment (bool): Whether to force alignment even if the ligand is large.
            alignment_energy_tolerance (float): Energy tolerance for RDKit alignment.
            seed (int): Random seed for reproducibility.
            verbose (bool): Whether to print verbose output.
            force_retry (bool): Whether to retry on failure.
            triangulate_dummies (bool): Whether to triangulate dummies in the ligand.
            pb_check (bool): Whether to use PoseBusters for checking the pose validity.
            random_rotation (bool): Whether to apply random rotation to the ligand coordinates.
        """
        super().__init__()

        # Warning for ignored kwargs
        if len(ignored_kwargs):
            print(f"[WARN] Ignored kwargs in SigmaDataset: {ignored_kwargs}. Please check for typos unless unintended.")
        
        # Datafront
        self.datafront = datafront

        # Parsing
        self.use_esm_embeddings = use_esm_embeddings
        self.esm_embeddings_clip_range = esm_embeddings_clip_range
        self.esm_embeddings_scaling_factor = esm_embeddings_scaling_factor
        self.fragmentation_strategy = fragmentation_strategy
        self.streamloading = streamloading
        # self.generator = torch.Generator().manual_seed(seed)
        self.seed = seed
        self.triangulate_dummies = triangulate_dummies
        self.ignore_triangulation = ignore_triangulation
        
        if ignore_triangulation:
            print("[WARN] Ignoring triangulation indexes in the dataset. This is only intended for Ablations.")
            
        if sample_conformer:
            print(
                "[INFO] Sample conformer is set to True. The bound ligand pose will NOT be used for fragmentation but rather a sampled conformation."  # noqa
            )
        self.sample_conformer = sample_conformer

        # Load ESM embeddings
        if self.use_esm_embeddings:
            esm_embeddings, esm_embeddings_idx = self.datafront.load_esm_embeddings()
            self.esm_embeddings = esm_embeddings
            self.esm_embeddings_idx = esm_embeddings_idx
        else:
            self.esm_embeddings = None
            self.esm_embeddings_idx = None

        # RDKit Alginment of Bound Pose & Fragmentation
        self.alignment_tries = alignment_tries
        if force_alignment and alignment_tries <= 0:
            print("[WARN] Force alignment is set to True but alignment_tries is 0. Setting alignment_tries to 5.")
            self.alignment_tries = 5
        self.alignment_rmsd_tolerance = alignment_rmsd_tolerance if alignment_rmsd_tolerance is not None else torch.inf
        self.force_alignment = force_alignment
        self.alignment_energy_tolerance = (
            alignment_energy_tolerance if alignment_energy_tolerance is not None else torch.inf
        )
        self.ignore_conjugated_torsion = ignore_conjugated_torsion

        # Protein Ligand Complex
        self.pocket_com_cutoff = pocket_com_cutoff
        self.pocket_com_noise = pocket_com_noise
        self.pocket_distance_cutoff = pocket_distance_cutoff
        self.pocket_distance_noise = pocket_distance_noise
        self.lig_coordinate_distance_noise = lig_coordinate_distance_noise
        self.prot_coordinate_distance_noise = prot_coordinate_distance_noise
        self.pocket_residue_outlier_factor = pocket_residue_outlier_factor
        self.keep_hetatoms = keep_hetatoms

        # Graph Construction
        self.pocket_virtual_cutoff = pocket_virtual_cutoff

        # Misc
        self.verbose = verbose
        self.skip_bounds_check = skip_bounds_check  # For inference
        self.retry = force_retry  # Only turn off in debugging when accessing the dataset directly.
        self.random_rotation = random_rotation
        self.mirror_prob = mirror_prob
        self.get_mol_info = get_mol_info

        # PoseBusters checker function (optional)
        if pb_check:
            try:
                self.pb_check = PoseBusters(
                    config="redock_fast",
                    max_workers=0,
                )
            except Exception as e:
                print(f"[WARN] Failed to initialize PoseBusters: {e}. Trying with 'redock' config.")
                try:
                    self.pb_check = PoseBusters(
                        config="redock",
                        max_workers=0,
                    )
                    # NOTE deleting internal energy from checks for speed.
                    energy_pb_config = self.pb_check.config["modules"][9]
                    assert energy_pb_config["name"] == "Energy ratio", (
                        f"PoseBusters config is not set to Energy. Please check the config: {energy_pb_config}."
                    )
                    # Remove the energy module from PoseBusters config to avoid unnecessary slow checks
                    del self.pb_check.config["modules"][9]
                except Exception as e:
                    print(f"[WARN] Failed to initialize PoseBusters with 'redock' config: {e}. Disabling PB checks.")
                    self.pb_check = None
        else:
            self.pb_check = None

    def get_bond_distance(self, atom1: str, atom2: str) -> float:
        return self.bond_lengths.get((atom1, atom2), self.bond_lengths.get((atom2, atom1), None))

    def parse_complex(
        self, sdf: str | Path, pdb: str | Path, ref_sdf: Optional[Path] = None
    ) -> dict[str, Chem.Mol | torch.Tensor | dict | None]:
        _ref_ligand_mol = None
        if ref_sdf is not None:
            pocket_struct, _ref_ligand_mol = sample_complex(
                sdf=ref_sdf,
                pdb=pdb,
                keep_hetatoms=self.keep_hetatoms,
                distance_cutoff=self.pocket_distance_cutoff,
                distance_noise=self.pocket_distance_noise,
                filter_outlier_factor=self.pocket_residue_outlier_factor,
            )
            # NOTE by default we remove Hs here unless strictly needed to preserve the original molecule.
            ligand_mols = read_ligands_from_sdf(sdf)
            assert len(ligand_mols) > 0, f"No valid ligands in {sdf}."
            ligand_mol = ligand_mols[0]
            ligand_coords_for_com = get_coordinates(_ref_ligand_mol, heavy_only=True)
        else:
            pocket_struct, ligand_mol = sample_complex(
                sdf=sdf,
                pdb=pdb,
                keep_hetatoms=self.keep_hetatoms,
                distance_cutoff=self.pocket_distance_cutoff,
                distance_noise=self.pocket_distance_noise,
                filter_outlier_factor=self.pocket_residue_outlier_factor,
            )
            ligand_coords_for_com = get_coordinates(ligand_mol, heavy_only=True)

        assert pocket_struct is not None, f"Failed to parse pocket from {pdb}."
        assert ligand_mol is not None, f"Failed to parse ligand from {sdf}."

        if isinstance(pocket_struct, str):
            pocket_mol: Chem.Mol = read_pdb_from_string(pocket_struct, as_biopython=False)
            pocket_struct = read_pdb_from_string(pocket_struct, as_biopython=True)
        elif isinstance(pocket_struct, PDB.Structure.Structure):
            pocket_mol = structure_to_rdkit(pocket_struct, remove_hs=False)
        else:
            raise ValueError(f"Unsupported pocket structure type: {type(pocket_struct)}")

        has_waters, has_hydrogens, has_hetatoms = inspect_structure(pocket_struct)
        assert not has_waters, f"Pocket structure {pdb} contains waters. Please remove them."
        assert not has_hydrogens, f"Pocket structure {pdb} contains hydrogens. Please remove them"
        assert not has_hetatoms, f"Pocket structure {pdb} contains heteroatoms. Please remove them"

        com = extract_pocket_com(
            pocket_struct,
            ligand_coords=ligand_coords_for_com,
            distance_cutoff=self.pocket_com_cutoff,
            keep_hetatoms=False,
        )
        return pocket_mol, ligand_mol, com, _ref_ligand_mol

    def fragment_and_annotate(self, mol: Chem.Mol, idx: int) -> tuple[Chem.Mol, dict]:
        # Fragment Ligand
        # TODO could pass a probability to ignore_conjugated_torsion here for data augmentation.
        frag_mol = fragment_molecule(
            mol,
            selection=self.fragmentation_strategy,
            ignore_conjugated=self.ignore_conjugated_torsion,
            verbose=self.verbose,
        )

        # NOTE Frag Map tells you the indices of torsional bonds (anchors) and the anchor2dummy mapping at the node (not the bond) # noqa
        frag_map = get_fragment_map(frag_mol)
        bond_indices = [b["bond"] for b in frag_map["torsional_bonds"]]
        bond_dofs = [b["dofs"] for b in frag_map["torsional_bonds"]]

        # Get the torsional bond lengths & ensure dummy-anchor mapping from frag_map reaches 0 bond lengths!
        frag_conf = frag_mol.GetConformer()
        torsional_coords = frag_conf.GetPositions()[bond_indices].astype(np.float32)  # [T, 2, 3] | []
        torsional_bond_lengths = np.zeros(len(bond_indices), dtype=np.float32)  # [T]
        if len(torsional_coords):
            torsional_bond_lengths = np.linalg.norm(
                torsional_coords[:, 0, :] - torsional_coords[:, 1, :], axis=1
            )  # [T]
            src = np.array(list(frag_map["anchor_to_dummy"].keys()))
            dst = list(frag_map["anchor_to_dummy"].values())

            src_stretch = np.repeat(src, np.array([len(x) for x in dst]))
            dst_stretch = np.concatenate(dst)
            # Assert mapping gives the correct 0-distance indices for the BCs.
            assert np.isclose(
                np.sum(frag_conf.GetPositions()[src_stretch] - frag_conf.GetPositions()[dst_stretch]), 0
            ), f"Fragmentation mapping is not correct. Check the mapping in molecule {idx}."

        # NOTE dummies here are the corresponding dummy of the anchor bond <A ----> D> not the dummy atom at the anchor coordinate as in frag_map. # noqa
        anchors, dummies = get_fragmented_anchors_dummies(frag_mol)
        anchors, dummies = np.array(anchors), np.array(dummies)

        # Find overconstrained anchors & dummies arising due to fragmentation merging.
        overconstrained_anchors = np.array(bond_indices)[np.array(np.equal(bond_dofs, 1))]
        overconstrained_dummies = dummies[[a in overconstrained_anchors for a in anchors]]

        # Create overconstraining dummies mask (which need to be ignored on ligand graph)
        # NOTE edge overconstraiend - overconstrained is what we want to remove!!! Not the dummy atom itself.
        # NOTE the dummy atom is still required if not fully overconstrained!!!
        # like Dof = [0,1], [1,0]. Can be removed if dof [1,1]
        mask = np.ones(frag_mol.GetNumAtoms(), dtype=bool)
        if len(overconstrained_dummies):
            mask[overconstrained_dummies] = False
        # In training naturally the dummies and anchors will ALWAYS meet but at inference... we can't assume
        # that the dummies will be at the same position as the anchors because the sampled conf. is different.

        # Get the bond indices of the dummies
        fragment_ids: list[list[int]] = get_fragments_as_mols(frag_mol, asMols=False)
        frag_sizes = [len(f) for f in fragment_ids]
        real_frag_sizes = torch.tensor([len([i for i in f if i not in overconstrained_dummies]) for f in fragment_ids])
        frag_idx = np.concatenate([np.repeat(i, x) for i, x in enumerate(frag_sizes)])
        frag_idx_mapping = np.array([frag_idx[i] for i in np.concatenate(fragment_ids)])

        # Virtual Nodes: (Num Frags x Num Virtual Nodes Per Frag)
        virtual_nodes: dict[int, list[dict]] = get_all_virtual_nodes_per_fragment(
            frag_mol, fragment_ids, atom_mask=mask
        )

        # Pivots & Fragment Anchors & Dummies
        anchor_bonds = [b["bond"] for b in frag_map["torsional_bonds"]]
        anchor_triangulation_indexes: dict = get_triangle_equality_mapping(
            frag_mol, get_torsional_neighbors(anchor_bonds)
        )
        triangulation_indexes = deepcopy(anchor_triangulation_indexes)

        # Get neighbor(s) for each anchor atom that is NOT part of the torsional bond.
        if self.triangulate_dummies:
            raise NotImplementedError("Triangulation of dummies is not implemented yet. This is a non-trivial problem because the dummy neighbors are not defined by the fragmentation but rather by the original molecule structure. We need to ensure that we are adding non-redundant triangulation edges for the dummies based on their original neighbors in the molecule. This requires careful handling of the mapping between anchors and dummies and their respective neighbors.")  # noqa: E501
        return (
            frag_mol,
            # First Dict has size [N] and [E]
            {
                "bond_lengths": torch.from_numpy(torsional_bond_lengths),
                "triangulation_indexes": [triangulation_indexes],
                "anchors": torch.from_numpy(anchors),
                "dummies": torch.from_numpy(dummies),
                "overconstrained_anchors": tensorise_idxs(overconstrained_anchors, frag_mol.GetNumAtoms()),
                "overconstrained_dummies": tensorise_idxs(overconstrained_dummies, frag_mol.GetNumAtoms()),
                "mask": torch.from_numpy(mask),
                "frag_atom_idx": torch.from_numpy(np.concatenate(fragment_ids)),
                "frag_counter": torch.from_numpy(frag_idx),
                "frag_idx_mapping": torch.from_numpy(frag_idx_mapping),
                "dummy_frag_sizes": torch.tensor(frag_sizes, dtype=torch.int32),
                "real_frag_sizes": torch.tensor(real_frag_sizes, dtype=torch.int32),
            },
            # Use these with care! They do not always batch.
            {
                "num_fragments": len(fragment_ids),
                "virtual_nodes": virtual_nodes,
                "fragment_ids": fragment_ids,
                "bond_indices": bond_indices,
                "bond_dofs": bond_dofs,
            },
        )

    def __len__(self) -> int:
        return len(self.datafront)

    def __getitem__(self, idx: int) -> Data:  # noqa
        # Get SDF & PDB files from idx
        lig_path, pdb_path, ref_sdf_path = self.datafront[idx]

        time_start = time.time()
        try:
            pocket, bound_ligand, pocket_com, ref_ligand_mol = self.parse_complex(
                sdf=lig_path,
                pdb=pdb_path,
                ref_sdf=ref_sdf_path,
            )

            # Jitter CoM for pocket according to perturbation (better pocket CoM coverage)
            pocket_com = pocket_com + np.random.normal(0, self.pocket_com_noise, size=pocket_com.shape).astype(
                np.float32
            )

            # Skip molecule if it has ridiculous number of torsional bonds
            num_torsionals = len(detect_torsional_bonds(bound_ligand, self.ignore_conjugated_torsion))
            if num_torsionals > MAX_TORSIONAL_BONDS and not self.skip_bounds_check:
                if self.verbose:
                    print(f"[WARN] Molecule {idx} has too many torsional bonds: {num_torsionals}.")
                raise ValueError(
                    f"Molecule {'/'.join(Path(lig_path).parts[-3:])} has too many torsional bonds: {num_torsionals}."
                )

            # ----------------  Protein ----------------
            # Protein
            pocket_graph = get_global_protein_graph(
                pocket,
                pdb_path,
                distance_cutoff=self.pocket_virtual_cutoff,
                esm_embeddings=self.esm_embeddings,
                esm_embeddings_idx=self.esm_embeddings_idx,
                esm_embeddings_clip_range=self.esm_embeddings_clip_range,
                esm_embeddings_scaling_factor=self.esm_embeddings_scaling_factor,
            )

            # ----------------  Ligand ----------------
            # RDKIT fragment distributional matching for starting bound ligand pose.
            aligned_ligand, rmsd, energy_delta = deepcopy(bound_ligand), 0.0, 0.0
            # Skip alignment if it is really big
            mol_weight = Chem.rdMolDescriptors.CalcExactMolWt(bound_ligand)
            alignment_tries = 0 if (mol_weight > MAX_WEIGHT and not self.force_alignment) else self.alignment_tries
            # Random alignment skip (p = 4/5) to also train on bound poses!
            if (alignment_tries > 0) and ((torch.rand(1).item() < 0.8) or self.force_alignment):
                optimizer = ConformerOptimizer(
                    aligned_ligand,
                    tries=alignment_tries,
                    tolerance=self.alignment_rmsd_tolerance,
                    max_energy_delta=self.alignment_energy_tolerance,
                    seed=torch.randint(0, 1000, [1]).item(),
                    ignore_conjugated=self.ignore_conjugated_torsion,
                    pb_check=self.pb_check,
                )
                aligned_mol, rmsd_, energy_delta_, pb_val_ = optimizer.optimize_torsions(pocket=pocket, strict=False)
                # Randomly care about PB invalid - this is a trick to make sure that the PB validation is not too strict.
                if self.pb_check is not None:
                    valid_enough = (
                        pb_val_
                        if pb_val_
                        else ((torch.rand(1).item() < 0.5) and (energy_delta_ < self.alignment_energy_tolerance))
                    )
                else:
                    valid_enough = energy_delta_ < self.alignment_energy_tolerance
                if rmsd_ < self.alignment_rmsd_tolerance and valid_enough:
                    aligned_ligand = aligned_mol
                    rmsd = rmsd_
                    energy_delta = energy_delta_
                else:
                    if self.verbose:
                        print(
                            f"[WARN] Ligand alignment failed with RMSD {rmsd_:.2f} > {self.alignment_rmsd_tolerance} \
                            or ENERGY {energy_delta_:.2f} > {self.alignment_energy_tolerance}. \
                            or PB-Val = {pb_val_}. \
                            Using bound pose for sample {idx}."
                        )
                    assert np.all(
                        aligned_ligand.GetConformer().GetPositions() == bound_ligand.GetConformer().GetPositions()
                    ), "Ligand conformer was modified in-place!"

            # NOTE aligned_ligand is the ligand with the same conformation as the bound ligand.
            # Fragment Ligand
            if self.sample_conformer:
                # This is for inference only
                optimizer = ConformerOptimizer(
                    deepcopy(bound_ligand),
                    seed=self.seed + idx,
                )
                sampled_conformation = optimizer._generate_conformer()
                frag_mol, frag_torchdata, frag_info = self.fragment_and_annotate(
                    mol=sampled_conformation,
                    idx=idx,
                )
            else:
                # This is for trianing
                frag_mol, frag_torchdata, frag_info = self.fragment_and_annotate(
                    mol=aligned_ligand,
                    idx=idx,
                )

            # NOTE this is only for ablations -> Always use these.
            if self.ignore_triangulation:
                frag_torchdata["triangulation_indexes"] = None

            # Build Ligand Graph & Add Virtual Nodes
            frag_graph = get_global_ligand_graph(frag_mol, **frag_torchdata, **frag_info)

            # Extend torchdata info with virtual node padding
            num_lig_virtual = sum([len(v) for k, v in frag_info["virtual_nodes"].items()])
            frag_torchdata["overconstrained_anchors"] = torch.cat(
                [frag_torchdata.get("overconstrained_anchors"), torch.zeros(num_lig_virtual)]
            )
            frag_torchdata["overconstrained_dummies"] = torch.cat(
                [frag_torchdata.get("overconstrained_dummies"), torch.zeros(num_lig_virtual)]
            )
            frag_torchdata["mask"] = torch.cat(
                [frag_torchdata.get("mask"), torch.ones(num_lig_virtual, dtype=torch.bool)]
            )

            # ------------ COMPLEX ------------
            # NOTE interaction graph swaps pos for ref_pos.
            interaction_graph = get_global_interaction_graph(
                protein_graph=pocket_graph,
                ligand_graph=frag_graph,
                lig_coordinate_noise=self.lig_coordinate_distance_noise,
                prot_coordinate_noise=self.prot_coordinate_distance_noise,
                random_rotation=self.random_rotation,  # Random rotation around the CoM for slight data augmentation (imperfect equivariance) # noqa
                pocket_com=pocket_com,  # Pocket CoM used if random_rotation is True.
            )

            time_end = time.time()
            data = Data(
                # Protein-Ligand Complex
                **interaction_graph,
                # Frags
                triangulation_indexes=frag_torchdata["triangulation_indexes"],
                overconstrained_anchors=frag_torchdata["overconstrained_anchors"],
                overconstrained_dummies=frag_torchdata["overconstrained_dummies"],
                frag_sizes=frag_torchdata["real_frag_sizes"],
                dummy_frag_sizes=frag_torchdata["dummy_frag_sizes"],
                # RDKIT MolData
                mol_info={
                    "mol_id": idx,
                    "original": bound_ligand,
                    "fragmented": frag_mol,
                    "aligned": aligned_ligand,
                    "reference": ref_ligand_mol,
                    "pocket": pocket,
                    "pdb_path": pdb_path,
                    "ligand_path": lig_path,
                    "num_torsional_bonds": num_torsionals,
                    "num_fragments": frag_info["num_fragments"],
                }
                if self.get_mol_info
                else None,
                # Misc
                process_time=torch.tensor(time_end - time_start, dtype=torch.float),
                alignment_rmsd=torch.tensor(rmsd, dtype=torch.float),
                alignment_energy_delta=torch.tensor(energy_delta, dtype=torch.float),
                pocket_com=torch.from_numpy(pocket_com).unsqueeze(0),
                seed=self.seed,
            )
            # Success
            return data

        except Exception as e:
            # Only print if Exception is not ValueError
            if not isinstance(e, ValueError):
                print(f"[WARN] Sample {idx} failed: {e}. Skipping...")
            if self.retry:
                return self._try_again()
            else:
                return None

    def _try_again(self) -> Data:
        # Optional logging for debugging
        idx = torch.randint(0, len(self), ()).item()
        if self.verbose:
            print(f"[INFO] Retrying sample {idx}...")
        return self.__getitem__(idx)

    def __repr__(self) -> str:
        return super().__repr__() + f"\n SigmaDock Dataset with {len(self)} points"


class SigmaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        # Datasets
        train_datafront: DataFront,
        val_datafront: DataFront | None = None,
        test_datafront: DataFront | None = None,
        # Misc
        batch_size: int = 64,
        num_workers: int = 4,
        persistent_workers: bool = True,
        seed: int = 42,
        # Cached Wrapper Dataset
        cache_factor: int = 2,
        cache_cycles: int = 4,
        val_cycles: int = 2,
        cache_strategy: Literal["random", "minimal_discrepancy"] = "random",
        dataset_augmentation_factor: int = 1,
        # -------> Protein Ligand Dataset KWargs <-------
        **structural_config: dict,
    ) -> None:
        """SigmaDock Data Module for Protein-Ligand Complexes.

        Args:
            train_datafront (DataFront): DataFront object containing the training dataset.
            val_datafront (DataFront, optional): DataFront object containing the validation dataset.
            test_datafront (DataFront, optional): DataFront object containing the test dataset.
            batch_size (int): Batch size for the dataloaders.
            num_workers (int): Number of workers for the dataloaders.
            persistent_workers (bool): Whether to use persistent workers for the dataloaders.
            seed (int): Random seed for reproducibility.
            cache_factor (int): Factor by which to increase the dataset size in the cache wrapper.
            cache_cycles (int): Number of cycles for caching in the cache wrapper.
            cache_val (bool): Whether to cache the validation dataset.
            cache_strategy (str): Strategy for caching. Options: "random", "minimal_discrepancy".
            structural_config (dict): Additional configuration for the SigmaDataset.
        """
        # Call the parent constructor
        super().__init__()

        # Datasets
        self.train_datafront = train_datafront
        self.val_datafront = val_datafront
        self.test_datafront = test_datafront
        self.dataset_augmentation_factor = dataset_augmentation_factor
        self.val_cycles = val_cycles

        # Loaders
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.seed = seed

        # Cache Wrapper
        self.cache_factor = cache_factor
        self.cache_cycles = cache_cycles
        self.cache_strategy = cache_strategy
        assert self.cache_strategy in ["random", "minimal_discrepancy"], (
            f"Cache strategy {self.cache_strategy} not supported."
        )

        # PL Kwargs
        self.structural_config: dict = structural_config
        # Save Hyperparameters
        self.save_hyperparameters(ignore=["train_datafront", "val_datafront", "test_datafront"])

    # Called on Master Rank only (GPU 0)
    def prepare_data(self) -> None:
        if getattr(self.structural_config, "use_esm_embeddings", False):
            print("[INFO] Preparing ESM3 embeddings.")
            # Filter datafronts to only include entries with ESM embeddings.
            if self.train_datafront is not None:
                self.train_datafront.filter_data_without_embeddings()
            if self.val_datafront is not None:
                self.val_datafront.filter_data_without_embeddings()
            if self.test_datafront is not None:
                self.test_datafront.filter_data_without_embeddings()

    # Called on every process (DDP)
    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # Create the datasets for train-val.
            train_dataset = SigmaDataset(
                datafront=self.train_datafront,
                **self.structural_config,
                seed=self.seed,
            )
            # Splits
            if self.val_datafront is None:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset,
                    [int(len(self.train_dataset) * 0.8), int(len(self.train_dataset) * 0.2)],
                    generator=torch.Generator().manual_seed(self.seed),
                )
            else:
                # Make val structural config with no noise.
                self.val_structural_config = deepcopy(self.structural_config)
                # Any key containing "noise" will be set to 0.0.
                for key in self.val_structural_config:
                    if "noise" in key:
                        print(f"[INFO] Setting {key} to 0.0 for validation dataset.")
                        self.val_structural_config[key] = 0.0
                val_dataset = SigmaDataset(
                    datafront=self.val_datafront,
                    **self.val_structural_config,
                    seed=self.seed,
                )
            # NOTE removing external data-caching for implicit data-caching -> Keeping footprint
            self.train_dataset = (
                CachedRecycleWrapper(
                    train_dataset,
                    batch_size=self.batch_size,
                    num_cycles=self.cache_cycles,
                    cache_factor=self.cache_factor,
                    dataset_len_augmentation_factor=self.dataset_augmentation_factor,
                )
                if self.cache_cycles > 1
                else train_dataset
                # IterableCachedRecycleWrapper(
                #     train_dataset,
                #     num_cycles=self.cache_cycles,
                #     cache_size=int(self.batch_size * self.cache_factor),
                #     seed=self.seed,
                # )
            )
            # NOTE removing external data-caching for implicit data-caching -> Keeping footprint
            self.val_dataset = (
                CachedRecycleWrapper(
                    val_dataset,
                    batch_size=self.batch_size,
                    cache_factor=1,
                    num_cycles=self.val_cycles,
                    # Spend more time on validation dataset for more accurate values (e.g. 4x)
                    dataset_len_augmentation_factor=self.val_cycles,
                )
                if (self.val_cycles > 1)
                else val_dataset
                # IterableDeterministicRecycleWrapper(
                #     val_dataset,
                #     cache_size=self.batch_size,
                #     num_cycles=self.val_cycles,
                # )
            )
        elif stage == "test":
            # Create the datasets for test.
            if self.test_datafront is None:
                raise ValueError("Test datafront is None. Please provide a test datafront.")
            self.test_dataset = SigmaDataset(
                datafront=self.test_datafront,
                **self.structural_config,
                seed=self.seed,
            )

    def train_dataloader(self) -> DataLoader:
        # NOTE Allow DDP to use DistributedSampler for training via Lightning Trainer for map-style instead of manually setting it.  # noqa: E501
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            sampler=None,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            worker_init_fn=worker_init_fn,
        )
        # NOTE: Deprecated for IterableDatasets.
        return CustomDataLoader(
            self.train_dataset,
            # For IterableDatasets, shuffle and sampler MUST be None.
            # The shuffling/sampling logic is handled inside __iter__.
            shuffle=False,
            sampler=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        # NOTE Allow DDP to use DistributedSampler for training via Lightning Trainer for map-style instead of manually setting it.  # noqa: E501
        return DataLoader(
            self.val_dataset,
            shuffle=True,
            sampler=None,
            pin_memory=True,
            # prefetch_factor=4,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            # Note constant epoch - same ordering every epoch for val set.
            worker_init_fn=worker_init_fn,
        )
        # NOTE: Deprecated for IterableDatasets.
        return CustomDataLoader(
            self.val_dataset,
            # For IterableDatasets, shuffle and sampler MUST be None.
            # The shuffling/sampling logic is handled inside __iter__.
            shuffle=False,
            sampler=None,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # drop_last=True,
            # collate_fn=custom_collate,
        )


if __name__ == "__main__":
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # Dataloaders
    from tqdm import tqdm

    from sigmadock.data import DataFront, SigmaDataset

    # Root paths
    ROOT = Path("/Users/alvaroprat/Desktop/")

    ASTEX_DATASET = ROOT / "SigmaDock/data/posebusters_paper_data/astex_diverse_set/"
    ASTEX_IDXS = ASTEX_DATASET.parent / "astex_diverse_set_ids.txt"
    assert ASTEX_DATASET.exists(), f"Path {ASTEX_DATASET} does not exist"
    assert ASTEX_IDXS.exists(), f"Path {ASTEX_IDXS} does not exist"

    POSEBUSTERS_DATASET = ROOT / "SigmaDock/data/posebusters_paper_data/posebusters_benchmark_set/"
    POSEBUSTERS_IDXS = POSEBUSTERS_DATASET.parent / "posebusters_benchmark_set_ids.txt"
    assert POSEBUSTERS_DATASET.exists(), f"Path {POSEBUSTERS_DATASET} does not exist"
    assert POSEBUSTERS_IDXS.exists(), f"Path {POSEBUSTERS_IDXS} does not exist"

    astex_df = pd.read_csv(ASTEX_IDXS, sep="\t", header=None)
    posebusters_df = pd.read_csv(POSEBUSTERS_IDXS, sep="\t", header=None)

    datafront = DataFront(
        POSEBUSTERS_DATASET,
        # Get raw protein .pdb
        pdb_regex=r"",
        # Get ligands file with multiple bound poses
        sdf_regex=r".*ligands.*\.sdf$",
    )

    dataset = SigmaDataset(
        datafront=datafront,
        pocket_com_cutoff=6.0,
        pocket_com_noise=0.5,
        pocket_distance_cutoff=6.0,
        pocket_distance_noise=0.5,
        pocket_residue_outlier_factor=2.0,
        streamloading=True,
        keep_hetatoms=False,
        alignment_tries=3,
        alignment_rmsd_tolerance=0.3,
        alignment_energy_tolerance=5.0,
        ignore_conjugated_torsion=False,
        seed=42,
    )

    # import os
    # num_workrers = os.cpu_count() - 1
    batch_size = 8
    num_workers = 0
    train_loader = DataLoader(
        CachedRecycleWrapper(
            dataset,
            batch_size=32,
            cache_factor=2,
            num_cycles=4,
            cache_strategy="random",
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers,
        pin_memory=True,
    )

    time_now = time.time()
    num_epochs = 4
    aligned_rmsds = []
    for epoch in range(num_epochs):
        time_pre_epoch = time.time()
        for batch in tqdm(train_loader):
            rmsds = batch.alignment_rmsd
            aligned_rmsds.append(rmsds)
        time_epoch = time.time() - time_pre_epoch
        print(f"Epoch {epoch}: {time_epoch:.2f} seconds")
    aligned_rmsds = torch.cat(aligned_rmsds)
    print(f"Mean RMSD: {aligned_rmsds.mean():.2f} +/- {aligned_rmsds.std():.2f}")
    average_batch_time = (time.time() - time_now) / (len(train_loader)) / num_epochs
    print(f"Average batch time: {average_batch_time:.2f} seconds")
