import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import posebusters
import spyrmsd
import spyrmsd.molecule
import spyrmsd.rmsd
import torch
from rdkit import Chem
from tqdm import tqdm

from sigmadock.chem.postprocessor import GNINA_METRICS

# TODO: for future releases, consider making this into an object-oriented class.
# TODO: could also return sorted idxs to return passing samples in the future?

# Define printable function with GLOBAL verbosity level
VERBOSE = False


def vprint(*args, **kwargs):  # noqa
    if VERBOSE:
        print(*args, **kwargs)


# ------- Constants -------


N_POSEBUSTERS = 428
PB_DICT_KEYS = [
    "mol_pred_loaded",
    "mol_true_loaded",
    "mol_cond_loaded",
    "sanitization",
    "inchi_convertible",
    "all_atoms_connected",
    "molecular_formula",
    "molecular_bonds",
    "double_bond_stereochemistry",
    "tetrahedral_chirality",
    "bond_lengths",
    "bond_angles",
    "internal_steric_clash",
    "aromatic_ring_flatness",
    "non-aromatic_ring_non-flatness",
    "double_bond_flatness",
    "internal_energy",
    "protein-ligand_maximum_distance",
    "minimum_distance_to_protein",
    "minimum_distance_to_organic_cofactors",
    "minimum_distance_to_inorganic_cofactors",
    "minimum_distance_to_waters",
    "volume_overlap_with_protein",
    "volume_overlap_with_organic_cofactors",
    "volume_overlap_with_inorganic_cofactors",
    "volume_overlap_with_waters",
]
N_ASTEX = 85


# ------- Compute Posebusters statistics -------


def get_mol_from_coords(coords: torch.Tensor, ref_mol: Chem.Mol) -> Chem.Mol:
    """Get a RDKit Mol object from given coordinates and a reference molecule."""
    copy_ref_mol = deepcopy(ref_mol)
    copy_ref_mol.RemoveAllConformers()
    conf = Chem.Conformer(copy_ref_mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords.tolist()):
        conf.SetAtomPosition(i, (x, y, z))
    _ = copy_ref_mol.AddConformer(conf)
    return copy_ref_mol


def compute_rmsd(mol1: Chem.Mol, mol2: Chem.Mol) -> tuple[float, float]:
    """Compute RMSD between two RDKit Mol objects using spyrmsd."""
    smol1 = spyrmsd.molecule.Molecule.from_rdkit(mol1)
    smol2 = spyrmsd.molecule.Molecule.from_rdkit(mol2)
    r_rmsd = spyrmsd.rmsd.rmsd(smol1.coordinates, smol2.coordinates, smol1.atomicnums, smol2.atomicnums)
    s_rmsd = spyrmsd.rmsd.rmsdwrapper(smol1, smol2)[0]
    return r_rmsd.item(), s_rmsd.item()


def compute_pb_checks(
    ligand: Chem.Mol,
    ref_ligand: Chem.Mol,
    ref_pocket: Chem.Mol,
    pbc: posebusters.PoseBusters,
) -> tuple[dict, bool]:
    """Compute PoseBusters checks for a given ligand pose
    against a reference ligand and pocket.

    Note that we return the average of all checks except RMSD.
    """

    df = pbc.bust(
        ligand,
        ref_ligand,
        ref_pocket,
        full_report=False,
    )
    df = df.copy()  # note sure why we need a copy?
    # NOTE: we exclude RMSD <= 2.0
    avg_pb_checks_without_rmsd = df.iloc[0][:-1].mean()
    pb_dict = df.iloc[0][:-1].to_dict()
    return pb_dict, avg_pb_checks_without_rmsd.item()


def compact_posebusting(results: dict[str, list[dict[str, Any]]], config: str = "redock") -> tuple[dict, dict, dict]:
    """Compute statistics (RMSD, PoseBusters checks, Posebusters dicts) for
    a single dictionary."""

    # We index by molecule ID and store a list of values (one per seed)
    all_rmsds: dict[int, float] = {}
    all_pb_checks: dict[int, float] = {}
    all_pb_dicts: dict[int, dict[str, bool]] = {}

    pbc_cache: dict[str, posebusters.PoseBusters] = {}

    def _pbc(name: str) -> posebusters.PoseBusters:
        if name not in pbc_cache:
            pbc_cache[name] = posebusters.PoseBusters(config=name, max_workers=0)
        return pbc_cache[name]

    for mol_id_i, per_mol_out in tqdm(results.items()):
        # Note it only does this for a SINGLE seed
        if len(per_mol_out) > 1:
            print(f"[WARNING] Expected 1 sample per mol_id_i={mol_id_i}, got {len(per_mol_out)}. Using first sample.")
        sample = per_mol_out[0]
        ref_lig = sample["lig_ref"]
        x0_hat = sample["x0_hat"]
        pred_lig = get_mol_from_coords(x0_hat, ref_lig)
        ref_pocket = sample["prot_ref"]

        # Cross-docking (separate reference SDF for pocket): use ``dock`` checks, not cognate ``redock`` identity tests.
        eff_config = "dock" if sample.get("crossdocking") else config
        pbc = _pbc(eff_config)

        _, s_rmsd = compute_rmsd(pred_lig, ref_lig)  # we only use srmsd
        pb_dict, avg_pb_checks = compute_pb_checks(pred_lig, ref_lig, ref_pocket, pbc)

        all_rmsds[mol_id_i] = s_rmsd
        all_pb_checks[mol_id_i] = avg_pb_checks
        all_pb_dicts[mol_id_i] = pb_dict

    return all_rmsds, all_pb_checks, all_pb_dicts


def compute_pb_statistics_per_path(output_dir_path: Path) -> tuple[dict, dict, dict]:
    """Compute statistics (RMSD, PoseBusters checks, Posebusters dicts) for
    a single seed/output directory path."""

    vprint(f"Computing Posebusters statistics for {output_dir_path}...")
    # Find output path where we assume each dir corresponds to a seed
    output_path = list(output_dir_path.glob("*predictions.pt"))
    assert len(output_path) == 1, f"Found {len(output_path)}!=1 prediction files in {output_dir_path}"
    output_path = output_path[0]

    # Load output dict and loop through protein-ligand pairs
    vprint(f"Loading output file: {output_path.name}")
    results: dict[str, torch.Tensor] = torch.load(output_path, weights_only=False)["results"]
    return compact_posebusting(results)


def compute_pb_statistics(
    output_dir_paths: list[Path],
    n_datatset: int,
    save: bool = False,
    max_workers: int = 0,
) -> tuple[dict]:
    """Compute statistics (RMSD, PoseBusters checks, Posebusters dicts) for
    multiple seeds in parallel."""

    # We save scores in the parent directory of output_dir_paths - i.e. model_id dir
    parent_paths = {path.parent for path in output_dir_paths}
    assert len(parent_paths) == 1, "All output_dir_paths should have the same parent directory"
    parent_path = next(iter(parent_paths))

    # Compute statistics in parallel over different seeds
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        statistics_list = executor.map(compute_pb_statistics_per_path, output_dir_paths)

    # Collect statistics from different seeds
    total_rmsds: dict[int, list[float]] = defaultdict(list)
    total_pb_checks: dict[int, list[float]] = defaultdict(list)

    def factory():  # noqa: ANN202
        return {key: [] for key in PB_DICT_KEYS}

    total_pb_dicts: dict[int, dict[str, list[bool]]] = defaultdict(factory)

    missing: dict[Path, list[int]] = defaultdict(list)

    for i, (all_rmsds, all_pb_checks, all_pb_dicts) in enumerate(statistics_list):
        assert len(all_rmsds) == len(all_pb_checks) == len(all_pb_dicts), "Mismatch in number of statistics"
        assert set(all_rmsds.keys()) == set(all_pb_checks.keys()) == set(all_pb_dicts.keys()), (
            "Mismatch in molecule IDs"
        )

        # Size of samples can vary due to dataloading issues
        for mol_id_i in range(n_datatset):
            # We fill in None for missing samples
            if mol_id_i not in all_rmsds:
                vprint(f"[WARNING] mol_id_i={mol_id_i} missing in {output_dir_paths[i]}, replacing with None")
                missing[output_dir_paths[i]].append(mol_id_i)

                total_rmsds[mol_id_i].append(None)
                total_pb_checks[mol_id_i].append(None)
                for key in PB_DICT_KEYS:
                    total_pb_dicts[mol_id_i][key].append(None)
            else:
                total_rmsds[mol_id_i].append(all_rmsds[mol_id_i])
                total_pb_checks[mol_id_i].append(all_pb_checks[mol_id_i])
                for key in PB_DICT_KEYS:
                    total_pb_dicts[mol_id_i][key].append(all_pb_dicts[mol_id_i][key])

    vprint(f"MISSING INDICES: {dict(missing)}")

    # Save aggregated statistics if needed
    if save:
        save_path = parent_path / "collected_posebusters_statistics.pt"
        torch.save(
            {
                "rmsds": dict(total_rmsds),
                "pb_checks": dict(total_pb_checks),
                "pb_dicts": dict(total_pb_dicts),
                "missing": dict(missing),
                "n_dataset": n_dataset,
            },
            save_path,
        )
        vprint(f"Saved Posebusters statistics to {save_path}")

    vprint("Finished computing Posebusters statistics.")

    return total_rmsds, total_pb_checks, total_pb_dicts, dict(missing)


# ------- Collection -------


def collect_scores_per_path(
    output_dir_path: Path, scoring: Literal["cnn", "vinardo", "vina"] = "vinardo"
) -> tuple[dict[int, dict[str, float]], dict[int, bool]]:
    """Collect affinity scores for a given output directory path."""

    assert scoring in ["vinardo", "cnn", "vina"], f"Unknown scoring method: {scoring}"
    vprint(f"Collecting scores ({scoring}) for {output_dir_path}...")

    # Find output path where we assume each dir corresponds to a seed
    output_path = list(output_dir_path.glob("*rescoring.pt"))
    assert len(output_path) == 1, f"Found {len(output_path)}!=1 score files ({scoring}) in {output_dir_path}"
    output_path = output_path[0]

    # Load output dict and loop through protein-ligand pairs
    scores = torch.load(output_path, weights_only=False)["scores"]
    all_scores: dict[int, dict[str, float]] = {}
    missing_scores: dict[int, bool] = defaultdict(lambda: False)
    for mol_id_i, per_mol_scores in scores.items():
        if len(per_mol_scores) == 0:
            vprint(f"[WARNING] mol_id_i={mol_id_i} has 0 scores; using empty dict.")
            missing_scores[mol_id_i] = True
            per_mol_scores = [dict.fromkeys(GNINA_METRICS)]
        elif len(per_mol_scores) != 1:
            missing_scores[mol_id_i] = False
            vprint(f"[WARNING] mol_id_i={mol_id_i} has {len(per_mol_scores)}!=1 multiple scores; using first score.")
        else:
            missing_scores[mol_id_i] = False
        score_dict = per_mol_scores[0]
        all_scores[mol_id_i] = score_dict
    return all_scores, missing_scores


def collect_scores(  # noqa: C901
    output_dir_paths: list[Path],
    dataset_keys: list[str],
    scoring: Literal["cnn", "vinardo", "vina"] = "vinardo",
    save: bool = False,
    verbose: bool = False,
) -> tuple[dict, dict, dict]:
    """Collect scores (i.e. vina, cnn (affinity), vinardo scores) for multiple seeds.

    Args:
        output_dir_paths: List of output directory paths (one per seed).
        dataset_keys: List of dataset keys (one per protein-ligand pair).
        scoring: Scoring method to use. One of "cnn", "vinardo", "vina".
        save: Whether to save the collected scores.
        verbose: Whether to print verbose output.
    Returns:
        total_scores: dict mapping mol_id_i to dict of lists of scores (one list per metric).
        all_missing_scores: dict mapping output_dir_path to list of mol_id_i with missing scores.
        missing: dict mapping output_dir_path to list of mol_id_i missing from sampling.

    NOTE:
    For either missing mol ids or missing scores, we fill in None.
    Therefore, for a given mol_id_i and seed, we either have None
    for both PB statistics and scores or a value for PB statistics
    and None for scores. For the former, we will resample from the the
    collection of valid seeds and for the latter, we will set the score
    to be the worst possible value.
    """
    if verbose:
        global VERBOSE
        VERBOSE = True
    else:
        VERBOSE = False
    vprint(f"Collecting scores using {scoring}...")

    if scoring not in ["vinardo", "cnn", "vina"]:
        vprint(f"[INFO] We do not need to collect scores for {scoring}, returning None.")
        return None, None, None

    # We save scores in the parent directory of output_dir_paths - i.e. model_id dir
    parent_paths = {path.parent for path in output_dir_paths}
    assert len(parent_paths) == 1, "All output_dir_paths should have the same parent directory"
    parent_path = next(iter(parent_paths))

    # Collect scores over different seeds
    collection = []
    for output_dir_path in output_dir_paths:
        collection.append(collect_scores_per_path(output_dir_path, scoring=scoring))

    # Aggregate scores from different seeds
    def factory():  # noqa: ANN202
        return {metric: [] for metric in GNINA_METRICS}

    total_scores: dict[int, dict[str, list[float]]] = defaultdict(factory)

    all_missing_scores: dict[Path, list[int]] = defaultdict(list)
    missing: dict[Path, list[int]] = defaultdict(list)

    for i, (all_scores, missing_scores) in enumerate(collection):
        for mol_key_i in dataset_keys:
            # We fill in None for missing samples due to dataloading issues
            if mol_key_i not in all_scores:
                vprint(f"[WARNING] Key={mol_key_i} missing in {output_dir_paths[i]}, replacing with None")
                missing[output_dir_paths[i]].append(mol_key_i)
                for metric in GNINA_METRICS:
                    total_scores[mol_key_i][metric].append(None)
            # We collect the elements with missing scores
            elif missing_scores[mol_key_i]:
                all_missing_scores[output_dir_paths[i]].append(mol_key_i)
                for metric in GNINA_METRICS:
                    total_scores[mol_key_i][metric].append(None)
            else:
                for metric in GNINA_METRICS:
                    total_scores[mol_key_i][metric].append(all_scores[mol_key_i][metric])

    vprint(f"MISSING SCORES: {dict(all_missing_scores)}")
    vprint(f"MISSING INDICES: {dict(missing)}")

    # Save aggregated scores if needed
    if save:
        save_path = parent_path / f"collected_scores_{scoring}.pt"
        torch.save(
            {
                "scores": dict(total_scores),
                "missing": dict(missing),
                "missing_scores": dict(all_missing_scores),
                "n_dataset": n_dataset,
            },
            save_path,
        )
        vprint(f"Saved collected scores to {save_path}")

    vprint("Finished collecting scores.")

    return dict(total_scores), dict(all_missing_scores), dict(missing)


def collect_posebusters_per_path(seed_path: Path) -> tuple[dict, dict, dict]:
    pb_file = seed_path / "posebusters.pt"
    if not pb_file.exists():
        vprint(f"[WARNING] Missing posebusters.pt in {seed_path}")
        return {}, {}, {}
    try:
        data = torch.load(pb_file)
        all_rmsds = data.get("rmsds", {})
        all_pb_checks = data.get("pb_checks", {})
        all_pb_dicts = data.get("pb_dicts", {})
        return all_rmsds, all_pb_checks, all_pb_dicts
    except Exception as e:
        vprint(f"[ERROR] Failed to load {pb_file}: {e}")
        return {}, {}, {}


def collect_posebusters(  # noqa: C901
    output_dir_paths: list[Path],
    dataset_keys: list[str],
    save: bool = False,
    verbose: bool = False,
) -> tuple[dict, dict, dict, dict]:
    """Load already-computed `posebusters.pt` from each seed folder and aggregate
    statistics (RMSD, PoseBusters checks, Posebusters dicts) across seeds.

    Args:
        output_dir_paths: List of seed directories (each expected to contain `posebusters.pt`).
        dataset_keys: List of dataset keys (one per protein-ligand pair), used to iterate and fill missing values.
        save: If True, save the aggregated results to the parent directory.
        verbose: If True, enable vprint-style verbose logging.

    Returns:
        total_rmsds: dict mapping mol_key -> list of rmsd values (one per seed; None for missing)
        total_pb_checks: dict mapping mol_key -> list of pb_check values (one per seed; None for missing)
        total_pb_dicts: dict mapping mol_key -> dict[key -> list[bool]] (one list per seed; None for missing)
        missing: dict mapping seed_path -> list of dataset_keys that were missing in that seed
    """
    # verbose switch (matches collect_scores pattern)
    if verbose:
        global VERBOSE
        VERBOSE = True
    else:
        VERBOSE = False

    # Ensure all seed folders share the same parent (model/run folder)
    parent_paths = {p.parent for p in output_dir_paths}
    assert len(parent_paths) == 1, "All output_dir_paths should have the same parent directory"
    parent_path = next(iter(parent_paths))

    # Load files sequentially
    statistics_list = []
    if len(output_dir_paths) == 0:
        # return empty structures with correct shapes
        empty_rmsds = defaultdict(list)
        empty_pb_checks = defaultdict(list)
        empty_pb_dicts = defaultdict(lambda: {k: [] for k in PB_DICT_KEYS})
        return empty_rmsds, empty_pb_checks, empty_pb_dicts, {}

    for p in output_dir_paths:
        statistics_list.append(collect_posebusters_per_path(p))

    # Aggregate results; maintain original output shapes
    total_rmsds: dict[str, list] = defaultdict(list)
    total_pb_checks: dict[str, list] = defaultdict(list)

    def factory():  # noqa: ANN202
        return {key: [] for key in PB_DICT_KEYS}

    total_pb_dicts: dict[str, dict[str, list]] = defaultdict(factory)
    missing: dict[Path, list[str]] = defaultdict(list)

    # statistics_list should align with output_dir_paths order
    for i, seed_path in enumerate(output_dir_paths):
        try:
            all_rmsds, all_pb_checks, all_pb_dicts = statistics_list[i]
        except IndexError:
            all_rmsds, all_pb_checks, all_pb_dicts = {}, {}, {}

        # defensive: ensure dict types
        if not (isinstance(all_rmsds, dict) and isinstance(all_pb_checks, dict) and isinstance(all_pb_dicts, dict)):
            vprint(f"[WARNING] unexpected data types for seed {seed_path}; treating as empty.")
            all_rmsds, all_pb_checks, all_pb_dicts = {}, {}, {}

        for mol_key in dataset_keys:
            # Fill in None for missing samples (either missing by sampling or I/O)
            if mol_key not in all_rmsds:
                vprint(f"[WARNING] Key={mol_key} missing in {seed_path}, replacing with None")
                missing[seed_path].append(mol_key)
                total_rmsds[mol_key].append(None)
                total_pb_checks[mol_key].append(None)
                for key in PB_DICT_KEYS:
                    total_pb_dicts[mol_key][key].append(None)
            else:
                # append rmsd and pb_checks; use .get for pb_checks defensively
                total_rmsds[mol_key].append(all_rmsds[mol_key])
                total_pb_checks[mol_key].append(all_pb_checks.get(mol_key))
                for key in PB_DICT_KEYS:
                    try:
                        total_pb_dicts[mol_key][key].append(all_pb_dicts[mol_key][key])
                    except Exception:
                        total_pb_dicts[mol_key][key].append(None)

    vprint(f"MISSING INDICES: {dict(missing)}")

    # Save aggregated statistics if requested
    if save:
        save_path = parent_path / "collected_posebusters_statistics.pt"
        torch.save(
            {
                "rmsds": dict(total_rmsds),
                "pb_checks": dict(total_pb_checks),
                "pb_dicts": dict(total_pb_dicts),
                "missing": dict(missing),
                "n_dataset": len(dataset_keys),
            },
            save_path,
        )
        vprint(f"Saved Posebusters statistics to {save_path}")

    vprint("Finished computing Posebusters statistics.")

    return dict(total_rmsds), dict(total_pb_checks), dict(total_pb_dicts), dict(missing)


# ------- Sort according to scores and compute top-k metrics -------


def clean_statistics_scores(  # noqa: C901
    rmsd: list[float | None],
    pb_checks: list[float | None],
    pb_dict: dict[str, list[bool]],
    scores: Optional[dict[str, list[float] | None]] = None,
    seed: int = 0,
    mol_key: str | None = None,
    score_used: str | None = None,
) -> tuple[list, list, dict, dict | None]:
    """
    Clean None values in statistics and scores by:
      - replacing fully-missing samples (rmsd None) by sampling a valid sample when available,
        otherwise by using per-field worst-value fallbacks;
      - replacing missing scores (but present rmsd) with per-key worst values.
    NOTE: This mutates the passed lists in-place and returns them.
    """

    # Basic consistency checks (same as original, but clearer names)
    n_rmsd = len(rmsd)
    n_pb_checks = len(pb_checks)
    lengths_pb = {len(v) for v in pb_dict.values()}
    if len(lengths_pb) != 1:
        raise AssertionError("Mismatch in number of seeds in pb_dict")
    n_pb = next(iter(lengths_pb))

    if scores is not None:
        lengths_scores = {len(v) for v in scores.values()}
        if len(lengths_scores) != 1:
            raise AssertionError("Mismatch in number of seeds in scores")
        n_scores = next(iter(lengths_scores))
    else:
        n_scores = n_rmsd

    if not (n_rmsd == n_pb_checks == n_pb == n_scores):
        raise AssertionError("Mismatch in number of seeds between rmsd/pb_checks/pb_dict/scores")

    random.seed(seed)

    # Indices for different problems
    failed_idxs: list[int] = []  # rmsd is None (and thus likely everything missing)
    non_score_idxs: list[int] = []  # rmsd present but one or more score keys None

    # helper to compute per-key fallback/worst values based on existing data
    def compute_worst_for_key(key: str, arr: list[float | None]) -> float:
        # gather non-None values
        vals = [v for v in arr if v is not None]
        if len(vals) > 0:
            # heuristics: for 'CNN' style keys lower is worse -> take min - margin
            if "cnn" in key.lower() or ("score" in key.lower() and "cnn" in key.lower()):
                base = min(vals)
                margin = abs(base) * 0.1 if base != 0 else 1.0
                return base - margin
            # for energies/affinity/higher-is-worse keys, take max + margin
            if any(tok in key.lower() for tok in ("affin", "energy", "intramol", "intramolecular")):
                base = max(vals)
                margin = abs(base) * 0.1 if base != 0 else 1.0
                return base + margin
            # fallback: assume higher is worse
            base = max(vals)
            margin = abs(base) * 0.1 if base != 0 else 1.0
            return base + margin
        # no data -> return conservative defaults by keyword, else generic large sentinel
        if "affin" in key.lower() or "energy" in key.lower():
            return 100.0
        if "cnn" in key.lower() or "score" in key.lower():
            return -100.0
        return 1e6

    # Build list of clean indices (rmsd not None, pb_dict consistent and scores present if available)
    for i, r in enumerate(rmsd):
        if r is None:
            failed_idxs.append(i)
            continue

        # check pb_dict consistency for this index
        pb_ok = all((key in pb_dict) and (i < len(pb_dict[key])) for key in pb_dict)
        if not pb_ok:
            # treat this like a failure of the whole sample
            failed_idxs.append(i)
            continue

        # check score presence if scores provided
        if scores is not None:
            if score_used is not None:
                missing_any = scores.get(score_used, [None] * n_rmsd)[i] is None
            else:
                missing_any = any(scores[key][i] is None for key in scores)
            if missing_any:
                non_score_idxs.append(i)

    # clean_idxs: indices with full valid data (rmsd not None and all scores non-None if scores provided)
    if scores is not None:
        clean_idxs = [i for i in range(n_rmsd) if i not in failed_idxs and i not in non_score_idxs]
    else:
        clean_idxs = [i for i in range(n_rmsd) if i not in failed_idxs]

    # If there are failed indices but no clean indices to sample from, use fallbacks
    if failed_idxs and not clean_idxs:
        print(f"[WARNING] No clean samples available to impute for {mol_key}. Using fallback worst-values.")
        # compute fallback values
        # rmsd fallback: use max existing rmsd or large sentinel
        existing_rmsd = [v for v in rmsd if v is not None]
        rmsd_fallback = max(existing_rmsd) if existing_rmsd else 1e6
        existing_pb = [v for v in pb_checks if v is not None]
        pb_fallback = max(existing_pb) if existing_pb else 1.0

        for i in failed_idxs:
            rmsd[i] = rmsd_fallback
            pb_checks[i] = pb_fallback
            for k in pb_dict:
                # pb dict are boolean checks; default False (failure)
                try:
                    pb_dict[k][i] = False
                except Exception:
                    # if pb_dict shorter, expand conservatively
                    # but we keep original length invariant in this function, so better raise
                    raise

            if scores is not None:
                for key in scores:
                    scores[key][i] = compute_worst_for_key(key, scores[key])

    # If we have clean indices, we can sample replacements for failed indices
    if failed_idxs and clean_idxs:
        replacement_idxs = [random.choice(clean_idxs) for _ in failed_idxs]
        for i, rep_i in zip(failed_idxs, replacement_idxs):
            # copy rmsd and pb_checks
            rmsd[i] = rmsd[rep_i]
            pb_checks[i] = pb_checks[rep_i]
            for key in pb_dict:
                pb_dict[key][i] = pb_dict[key][rep_i]
            if scores is not None:
                for key in scores:
                    scores[key][i] = scores[key][rep_i]

    # Now fix non_score_idxs: these have valid rmsd but missing one or more score keys
    if scores is not None and non_score_idxs:
        for i in non_score_idxs:
            for key in scores:
                if scores[key][i] is None:
                    scores[key][i] = compute_worst_for_key(key, scores[key])

    return rmsd, pb_checks, pb_dict, scores


def compute_heuristic(  # noqa: C901
    scores: dict[str, list[float]],
    scoring: str | None,
    scoring_config: dict,
    seed: int = 0,
) -> tuple[list[int], list[float]]:
    """Compute ordering (sorted indices) and the used_scores for one molecule.

    Behavior:
    - If ``scoring`` is ``"vinardo"`` or ``"cnn"`` the function expects
      ``scoring_config["score_name"]`` to exist and will sort by
      ``scores[score_name]`` (ascending/descending depending on the name).
    - If ``scoring`` == "pb" the function expects ``scoring_config["pb_checks"]``
      to be a list of keys that exist inside ``scores``; it computes the
      average across those keys (per-sample) and sorts descending.
    - If ``scoring`` == "heuristic" the function will compute a mixed score
      equal to the (direction-normalised) base scoring value multiplied by
      the average of the provided PB checks. This implements:
      "heuristic = scoring_score * avg(pb_checks)".
      The function therefore expects both ``score_name`` (which names the
      scoring key inside ``scores``) and ``checks`` (list of PB check keys)
      in ``scoring_config``.
    - If ``scoring`` is None, returns a random permutation of indices.

    Returns
    -------
    sorted_idxs:
        list of indices (ints) in the order to be used for reordering other
        statistics.
    used_scores:
        The score values (floats) in the same order as sorted_idxs.
    """

    if scoring is None:
        # Random ordering
        if not scores:
            return [], []
        n = len(next(iter(scores.values())))
        idxs = list(range(n))
        random.seed(seed)
        perm = random.sample(idxs, len(idxs))
        return perm, [None] * len(perm)

    # scoring is provided -> we need scores to exist
    assert scores is not None and scores, "scores must be provided when scoring is set"

    if scoring in ("vinardo", "cnn"):
        assert "score_name" in scoring_config, "score_name must be in scoring_config"
        score_name = scoring_config["score_name"]
        if score_name not in scores:
            raise KeyError(f"score_name {score_name} not found in scores")
        # Determine sort direction based on common score semantics
        if score_name in ["Affinity", "Intramolecular energy"]:
            reverse = False
        elif score_name in ["CNNscore"]:
            reverse = True
        else:
            # Default: descending (higher is better)
            reverse = True
        seq = list(enumerate(scores[score_name]))
        seq.sort(key=lambda x: x[1], reverse=reverse)
        sorted_idxs = [i for i, _ in seq]
        used_scores = [s for _, s in seq]
        return sorted_idxs, used_scores

    if scoring == "pb":
        assert "pb_checks" in scoring_config, "checks must be provided for pb scoring"
        checks = scoring_config["pb_checks"]
        n = len(next(iter(scores.values())))
        avg_check = []
        for i in range(n):
            vals = [scores[check][i] for check in checks]
            avg_check.append(sum(vals) / len(vals))
        seq = list(enumerate(avg_check))
        seq.sort(key=lambda x: x[1], reverse=True)
        sorted_idxs = [i for i, _ in seq]
        used_scores = [s for _, s in seq]
        return sorted_idxs, used_scores

    if scoring == "heuristic":
        # Implement heuristic = (direction-normalised scoring value) * avg(pb_checks)
        assert "score_name" in scoring_config, "score_name must be provided for heuristic mixing"
        assert "pb_checks" in scoring_config, "checks must be provided for heuristic mixing"
        score_name = scoring_config["score_name"]
        checks = scoring_config["pb_checks"]

        if score_name not in scores:
            raise KeyError(f"score_name {score_name} not found in scores for heuristic")
        for c in checks:
            if c not in scores:
                raise KeyError(f"PB check {c} not found in scores for heuristic")

        base = list(scores[score_name])
        # Normalize direction so that "higher is better" for the base score
        if score_name in ["Affinity", "Intramolecular energy"]:
            base = [-s for s in base]

        n = len(base)
        avg_pb = []
        for i in range(n):
            vals = [scores[check][i] for check in checks]
            avg_pb.append(sum(vals) / len(vals))

        # Mixed score is the product
        mixed = [
            b * (scoring_config["score_bias"] + p ** (scoring_config["pb_exponent"])) for b, p in zip(base, avg_pb)
        ]

        seq = list(enumerate(mixed))
        seq.sort(key=lambda x: x[1], reverse=True)
        sorted_idxs = [i for i, _ in seq]
        used_scores = [s for _, s in seq]
        return sorted_idxs, used_scores

    raise ValueError(f"Unknown scoring method: {scoring}")


def sort_statistics_for_top_k(
    total_rmsds: dict[int, list[float]],
    total_pb_checks: dict[int, list[float]],
    total_pd_dicts: dict[int, dict[str, list[bool]]],
    total_scores: dict[int, dict[str, list[float]]],
    dataset_keys: list[str],
    output_dir_paths: list[Path] | None = None,
    save: bool = False,
    seed: int = 0,
    scoring: str | None = None,
    **scoring_config: dict,
) -> tuple[
    dict[int, list[float]],
    dict[int, list[float]],
    dict[int, dict[str, list[bool]]],
    dict[int, dict[str, list[float]]],
    dict[int, list[float]],
]:
    """Sort statistics per protein-ligand pair based on scoring method.

    This version expects `total_scores` to be provided (unless `scoring` is None
    in which case a random ordering is used). The actual logic that computes the
    ordering for each molecule is delegated to `compute_heuristic`.
    """

    vprint = globals().get("vprint", print)
    vprint(f"Sorting statistics using {scoring} and seed={seed}...")

    tag = str(scoring)
    if scoring is None:
        tag = str(None)
    else:
        # try to build a descriptive tag when possible
        if scoring in ("vinardo", "cnn") and "score_name" in scoring_config:
            tag = f"{scoring}_{scoring_config['score_name']}"
        elif scoring == "heuristic":
            tag = f"heuristic_{scoring_config.get('score_name', 'unknown')}_with_{'_'.join(scoring_config.get('checks', []))}"  # noqa: E501
        elif scoring == "pb" and "pb_checks" in scoring_config:
            tag = f"pb_with_{'_'.join(scoring_config.get('pb_checks', []))}"
        else:
            raise ValueError(f"Cannot build descriptive tag for scoring={scoring} with config {scoring_config}")

    total_sorted_rmsds: dict[int, list[float]] = {}
    total_sorted_pb_checks: dict[int, list[float]] = {}
    total_sorted_pb_dicts: dict[int, dict[str, list[bool]]] = {}
    total_sorted_scores: dict[int, dict[str, list[float]]] = {}
    total_used_scores: dict[int, list[float]] = {}

    for mol_key_i in dataset_keys:
        rmsd = total_rmsds[mol_key_i]
        pb_checks = total_pb_checks[mol_key_i]
        pb_dict = total_pd_dicts[mol_key_i]
        scores = total_scores.get(mol_key_i, {}) if total_scores is not None else {}

        # Clean None values by random replacement (assumes this helper exists)
        rmsd, pb_checks, pb_dict, scores = clean_statistics_scores(
            rmsd,
            pb_checks,
            pb_dict,
            scores=scores,
            seed=seed,
            mol_key=mol_key_i,
            score_used=scoring_config.get("score_name"),
        )

        if scoring is None:
            sorted_idxs, used_scores = compute_heuristic(scores, None, {}, seed=seed)
        else:
            # Add PB Checks (chosen) to scoring_config if not already present
            flattened_pb_checks = {key: pb_dict[key] for key in scoring_config.get("pb_checks", []) if key in pb_dict}
            scores.update(flattened_pb_checks)
            sorted_idxs, used_scores = compute_heuristic(scores, scoring, scoring_config, seed=seed)

        total_sorted_rmsds[mol_key_i] = [rmsd[i] for i in sorted_idxs]
        total_sorted_pb_checks[mol_key_i] = [pb_checks[i] for i in sorted_idxs]
        total_sorted_pb_dicts[mol_key_i] = {key: [pb_dict[key][i] for i in sorted_idxs] for key in pb_dict}

        if scores:
            total_sorted_scores[mol_key_i] = {key: [scores[key][i] for i in sorted_idxs] for key in scores}
        else:
            total_sorted_scores[mol_key_i] = {}

        total_used_scores[mol_key_i] = used_scores

    if save:
        assert output_dir_paths is not None, "output_dir_paths must be provided if save is True"
        parent_paths = {path.parent for path in output_dir_paths}
        assert len(parent_paths) == 1, "All output_dir_paths should have the same parent directory"
        parent_path = next(iter(parent_paths))
        save_path = parent_path / f"sorted_statistics_and_scores_{tag}_seed_{seed}.pt"
        torch.save(
            {
                "rmsds": total_sorted_rmsds,
                "pb_checks": total_sorted_pb_checks,
                "pb_dicts": total_sorted_pb_dicts,
                "scores": total_sorted_scores,
                "used_scores": total_used_scores,
                "scoring": scoring,
                "scoring_config": scoring_config,
                "seed": seed,
                "n_dataset": n_dataset,
            },
            save_path,
        )
        vprint(f"Saving sorted statistics and scores to {save_path}")

    vprint("Finished sorting statistics")

    return (
        total_sorted_rmsds,
        total_sorted_pb_checks,
        total_sorted_pb_dicts,
        total_sorted_scores,
        total_used_scores,
    )


def compute_top_k_statistics(  # noqa: C901
    total_sorted_rmsds: dict,
    total_sorted_pb_checks: dict,
    dataset_keys: list[str],
    N: int | None = None,
    seed: int = 0,
    ks: list[int] | None = None,
    rmsd_thresholds: list[float] | None = None,
    shuffle_for_ties: bool = False,
    total_used_scores: dict | None = None,
) -> tuple[dict, dict, list[int]]:
    """Compute top-k statistics based on sorted RMSDs and PoseBusters checks."""
    if rmsd_thresholds is None:
        rmsd_thresholds = [2.0]
    if ks is None:
        ks = [1]
    if rmsd_thresholds is None:
        rmsd_thresholds = [2.0]
    assert set(total_sorted_rmsds.keys()) == set(total_sorted_pb_checks.keys()), "Mismatch in molecule IDs"
    if N is None:
        N = len(next(iter(total_sorted_rmsds.values())))
    # Set seed for shuffling which N sample seeds to consider
    num_seeds = len(next(iter(total_sorted_rmsds.values())))
    assert max(ks) <= N, f"max(ks)={max(ks)} must be <= N={N}"
    assert num_seeds >= N, f"N={N} must be <= number of seeds={num_seeds}"
    vprint(f"Computing top-k statistics with N={N}/{num_seeds} sampled seeds (using random seed={seed})...")
    vprint(f"Shuffling ties is set to {shuffle_for_ties}")
    idxs = list(range(num_seeds))

    # Loop through each molecule and compute top-k success rates
    passes = {k: {thr: [] for thr in rmsd_thresholds} for k in ks}
    passes_with_pb = {k: {thr: [] for thr in rmsd_thresholds} for k in ks}

    # for mol_id_i in total_sorted_rmsds.keys():
    for i, mol_key_i in enumerate(dataset_keys):
        random.seed(seed + i)
        shuffled_idxs = random.sample(idxs, len(idxs))
        N_idxs = shuffled_idxs[:N]
        N_idxs.sort(key=lambda x: x)  # ascending order list [N]
        sorted_rmsd = total_sorted_rmsds[mol_key_i]
        sorted_pb_checks = total_sorted_pb_checks[mol_key_i]

        # We only consider N samples after shuffling - this preserves score ordering
        sorted_rmsd = [sorted_rmsd[i] for i in N_idxs]
        sorted_pb_checks = [sorted_pb_checks[i] for i in N_idxs]

        # Deal with ties here
        if shuffle_for_ties:
            assert total_used_scores is not None, "total_used_scores must be provided if shuffle_for_ties is True"
            used_scores = total_used_scores[mol_key_i]
            used_scores = [used_scores[i] for i in N_idxs]
            indices_by_value = defaultdict(list)
            for i, value in enumerate(used_scores):
                indices_by_value[value].append(i)
            new_index_map = [0] * len(used_scores)
            for value, original_indices in indices_by_value.items():  # noqa: B007
                # Create a shuffled copy of the indices for the new positions
                shuffled_indices = original_indices[:]
                random.shuffle(shuffled_indices)
                # Map each original index to its new shuffled index
                for original_idx, new_idx in zip(original_indices, shuffled_indices):
                    new_index_map[original_idx] = new_idx
            sorted_rmsd = [sorted_rmsd[i] for i in new_index_map]
            sorted_pb_checks = [sorted_pb_checks[i] for i in new_index_map]
            _used_scores = [used_scores[i] for i in new_index_map]
            assert _used_scores == used_scores, "used_scores should remain the same after shuffling ties"

        for k in ks:
            top_k_rmsd = sorted_rmsd[:k]
            top_k_pb_checks = sorted_pb_checks[:k]
            for thr in rmsd_thresholds:
                pass_k = any(r <= thr for r in top_k_rmsd)
                pass_k_with_pb = any((r <= thr) and (pb >= 1.0) for r, pb in zip(top_k_rmsd, top_k_pb_checks))
                passes[k][thr].append(pass_k)
                passes_with_pb[k][thr].append(pass_k_with_pb)

    # Compute average success rates
    avg_passes = {k: {thr: sum(passes[k][thr]) / len(passes[k][thr]) for thr in rmsd_thresholds} for k in ks}
    avg_passes_with_pb = {
        k: {thr: sum(passes_with_pb[k][thr]) / len(passes_with_pb[k][thr]) for thr in rmsd_thresholds} for k in ks
    }

    vprint("Finished computing top-k statistics")

    return avg_passes, avg_passes_with_pb, N_idxs


def run_permutation_topk(
    total_rmsds: dict,
    total_pb_checks: dict,
    total_pd_dicts: dict,
    total_scores: dict,
    dataset_keys: list[str],
    *,
    num_permutations: int = 20,
    ks: list[int],
    rmsd_thresholds: list[float],
    N_max: int,
    sort_seed: int = 0,
    shuffle_for_ties: bool = True,
    scoring: str = "vinardo",
    scoring_config: dict | None = None,
    seed_pool: list[int] | None = None,
    progress: bool = True,
) -> dict:
    """
    Run `num_permutations` permutation experiments that call sorting + top-k
    statistic code and collect results.

    Parameters
    ----------
    total_rmsds, total_pb_checks, total_pd_dicts, total_scores : dict
        Inputs passed to sort_statistics_for_top_k.
    dataset_keys : list[str]
        List of dataset keys passed to compute_top_k_statistics.
    num_permutations : int
        Number of permutation runs to perform.
    ks : list[int]
        k values to evaluate.
    rmsd_thresholds : list[float]
        RMSD thresholds to evaluate.
    N_max : int
        Value passed as N to compute_top_k_statistics.
    sort_seed : int
        Seed passed into sort_statistics_for_top_k (deterministic sorting).
    shuffle_for_ties : bool
        Passed to compute_top_k_statistics.
    scoring : str
        Scoring method passed to sort_statistics_for_top_k.
    scoring_config : dict | None
        Extra keyword args forwarded to sort_statistics_for_top_k (can be None).
    seed_pool : list[int] | None
        If provided, uses these seeds for the permutations (length should be num_permutations).
        If None, seeds are drawn randomly from [0, 1000).
    progress : bool
        Whether to show tqdm progress bar.

    Returns
    -------
    dict
        all_results with keys:
            'success_rates' : list of avg_passes (one per permutation)
            'success_rates_with_pb' : list of avg_passes_with_pb
            'seed_indices' : list of N_idxs (seed indices used inside compute_top_k_statistics)
    """
    if scoring_config is None:
        scoring_config = {}

    # Prepare seeds
    if seed_pool is None:
        seed_pool = np.random.randint(0, 1000, size=num_permutations).tolist()
    else:
        if len(seed_pool) != num_permutations:
            raise ValueError("seed_pool length must equal num_permutations")

    all_results: dict = {"success_rates": [], "success_rates_with_pb": [], "seed_indices": []}

    iterator = seed_pool
    if progress:
        iterator = tqdm(seed_pool)

    for N_seed in iterator:
        # Call the sorting function (user-provided)
        tmp_sorted = sort_statistics_for_top_k(
            total_rmsds=total_rmsds,
            total_pb_checks=total_pb_checks,
            total_pd_dicts=total_pd_dicts,
            total_scores=total_scores,
            dataset_keys=dataset_keys,
            seed=sort_seed,
            scoring=scoring,
            **(scoring_config or {}),
        )
        (
            total_sorted_rmsds,
            total_sorted_pb_checks,
            total_sorted_pb_dicts,
            total_sorted_scores,
            total_used_scores,
        ) = tmp_sorted

        # Compute top-k statistics using the sampled N_seed
        avg_passes, avg_passes_with_pb, N_idxs = compute_top_k_statistics(
            total_sorted_rmsds=total_sorted_rmsds,
            total_sorted_pb_checks=total_sorted_pb_checks,
            dataset_keys=dataset_keys,
            N=N_max,
            seed=N_seed,
            ks=ks,
            rmsd_thresholds=rmsd_thresholds,
            shuffle_for_ties=shuffle_for_ties,
            total_used_scores=total_used_scores,
        )

        res = {
            "success_rates": avg_passes,
            "success_rates_with_pb": avg_passes_with_pb,
            "seed_indices": N_idxs,
        }
        for k in all_results:
            all_results[k].append(res[k])

    return all_results


if __name__ == "__main__":
    # Define which output directories to process
    output_dir_paths = []

    SAMPLES_PATH = Path("path/to/samples")
    exp_name = "posebusters"
    model_id = "model_id"
    n_dataset = N_POSEBUSTERS  # NOTE: important for None issue from dataloader

    for i in range(1, 41):
        run_tag = f"{model_id}_seed_{i}"
        path = SAMPLES_PATH / exp_name / model_id / run_tag
        output_dir_paths.append(path)

    # Config
    scoring = "vinardo"  # "vinardo", "cnn", "heuristic", None
    save = True
    max_workers = 5

    # Collect and save PB statistics
    total_rmsds, total_pb_checks, total_pb_dicts, missing = compute_pb_statistics(
        output_dir_paths, n_dataset, save=save, max_workers=max_workers
    )
