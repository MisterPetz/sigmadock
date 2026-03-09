import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from rdkit import Chem
from tqdm import tqdm

GNINA_METRICS = ["Affinity", "CNNscore", "CNNaffinity", "CNNvariance", "Intramolecular energy"]
ROOT_DIR = Path(__file__).resolve().parents[3]


def prep_ligand(ligand: Chem.Mol, tmp_path: Path, preprocess: bool = False) -> Path:
    """Convert and preprocess ligand to a temporary SDF file."""
    if preprocess:
        ligand = Chem.AddHs(ligand, addCoords=True)
    mol_block = Chem.MolToMolBlock(ligand)
    with open(tmp_path, "w") as f:
        f.write(mol_block)
    return tmp_path


def prep_protein(pdb_path: Path, tmp_path: Path, preprocess: bool = False) -> Path:
    if preprocess:
        try:
            from openmm.app import PDBFile # type: ignore # noqa
            from pdbfixer import PDBFixer # type: ignore
        except ImportError as e:
            raise ImportError(
                "pdbfixer and openmm are required for protein preprocessing. "
                "Please install them via 'pip install pdbfixer openmm'."
            ) from e
        fixer = PDBFixer(filename=str(pdb_path))
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(False)  # false also removes water
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        with open(tmp_path, "w", encoding="utf-8") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return tmp_path
    else:
        return pdb_path


def parse_gnina_output(output: str) -> dict[str, float]:
    """Parse gnina output string to extract scoring metrics."""
    score_dict = dict.fromkeys(GNINA_METRICS)
    lines = output.strip().split("\n")
    for line in lines:
        for metric in GNINA_METRICS:
            if line.startswith(metric):
                score = float(line.split(":")[1].split()[0])
                score_dict[metric] = score
                break
    return score_dict


def gnina_score(
    ligand: Chem.Mol,
    protein_pdb_path: Path,
    tmp_ligand_path: Path,
    tmp_pdb_path: Path,
    scoring: str = "vina",
    # config
    preprocess: bool = False,
    device: int = 0,
    no_gpu: bool = False,
) -> dict[str, float]:
    """Compute gnina score for a given ligand and protein."""
    ligand_path = prep_ligand(ligand, tmp_ligand_path, preprocess=preprocess)
    protein_path = prep_protein(protein_pdb_path, tmp_pdb_path, preprocess=preprocess)

    # Call gnina with the prepared paths
    command = [
        "gnina",
        "-r",
        str(protein_path),
        "-l",
        str(ligand_path),
        "--autobox_ligand",
        str(ligand_path),
        "--score_only",
    ]

    if no_gpu:
        command += ["--no_gpu"]
    else:
        command += ["--device", str(device)]  # FIXEME: might be bug here with argsparse

    if scoring == "vinardo" or scoring == "vina":
        command += ["--scoring", scoring, "--cnn_scoring", "none"]
    elif scoring == "cnn":
        command += ["--cnn_scoring", "all"]
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")

    # Necessary for multi-rank contention
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse the output
    output = result.stdout
    score_dict = parse_gnina_output(output)

    return score_dict


def get_mol_from_coords(coords: torch.Tensor, ref_mol: Chem.Mol) -> Chem.Mol:
    """Get a RDKit Mol object from given coordinates and a reference molecule."""
    copy_ref_mol = deepcopy(ref_mol)
    copy_ref_mol.RemoveAllConformers()
    conf = Chem.Conformer(copy_ref_mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords.tolist()):
        conf.SetAtomPosition(i, (x, y, z))
    _ = copy_ref_mol.AddConformer(conf)
    return copy_ref_mol


def compute_gnina_score(
    outputs_path: Path, scoring: str = "cnn", **gnina_config: dict
) -> tuple[dict[int, list[dict]], list[tuple[int, int, str]]]:
    """
    Compute gnina scores for a single prediction file (outputs_path).
    Returns (scores, failed) where:
      - scores: mapping mol_id -> list of score dicts
      - failed: list of tuples (mol_id, idx, pdb_stem) for failures
    Does NOT iterate over the directory; only processes the given file.
    """
    print(f"Processing outputs file: {outputs_path.name}")

    scores: dict[int, list[dict]] = {}
    failed: list[tuple[int, int, str]] = []

    try:
        loaded = torch.load(outputs_path, weights_only=False)
        outputs = loaded["results"]
    except Exception as e:
        print(f"[ERROR] Failed to load {outputs_path}: {e}")
        # Mark the whole file as failed with a sentinel entry
        failed.append((-1, -1, outputs_path.name))
        return scores, failed

    for mol_id_i, per_mol_out in tqdm(outputs.items()):
        # print(f"Processing mol_id_i={mol_id_i} with {len(per_mol_out)} samples")
        if len(per_mol_out) != 1:
            print(f"[WARNING] Multiple samples for mol_id_i={mol_id_i}.")
        scores[mol_id_i] = []

        for idx, sample in enumerate(per_mol_out):
            ref_lig = sample["lig_ref"]
            ref_pdb_path = sample["pdb_path"]
            # print(f"pdb_file: {ref_pdb_path.name}")

            x0_hat = sample["x0_hat"]
            pred_lig = get_mol_from_coords(x0_hat, ref_lig)

            tmp_ligand_path = ref_pdb_path.parent / (ref_pdb_path.stem + "_ligand_tmp.sdf")
            tmp_pdb_path = ref_pdb_path.parent / (ref_pdb_path.stem + "_protein_tmp.pdb")

            try:
                with tempfile.TemporaryDirectory(prefix="gnina_") as tmpdir:
                    tmp_ligand_path = Path(tmpdir) / (ref_pdb_path.stem + "_ligand_tmp.sdf")
                    tmp_pdb_path = Path(tmpdir) / (ref_pdb_path.stem + "_protein_tmp.pdb")

                    # Call gnina_score which should write/read these tmp paths
                    score_dict = gnina_score(
                        pred_lig,
                        ref_pdb_path,
                        tmp_ligand_path,
                        tmp_pdb_path,
                        scoring=scoring,
                        **gnina_config,
                    )
                    scores[mol_id_i].append(score_dict)

            except Exception as e:
                # include exception text to aid debugging
                print(
                    f"[ERROR] Failed to process mol_id_i={mol_id_i}, idx={idx}, ref_pdb_path={ref_pdb_path.stem}: {e}"
                )
                failed.append((mol_id_i, idx, f"{ref_pdb_path.stem}: {e}"))

    print(f"Finished processing file: {outputs_path.name}")
    return scores, failed


def gnina_score_for_outputs(outputs_dir_path: Path, scoring: str = "cnn", **gnina_config: dict) -> dict:
    """
    Iterate over outputs_dir_path, call compute_gnina_score on each
    *_predictions.pt file, save per-file rescored outputs, and return
    a mapping of filename -> failed-list.
    """
    print(f"gnina scoring for {outputs_dir_path}")
    all_failed: dict = {}

    for outputs_path in outputs_dir_path.iterdir():
        if not outputs_path.name.endswith("_predictions.pt"):
            continue

        scores, failed = compute_gnina_score(outputs_path, scoring=scoring, **gnina_config)

        # Save gnina scores for outputs_path
        name = outputs_path.stem + f"_rescored_{scoring}.pt"
        save_path = outputs_dir_path / name
        torch.save({"scores": scores, "failed": failed, "gnina_config": gnina_config}, save_path)
        print(f"Saved rescored results to: {save_path}")

        all_failed[outputs_path.name] = failed

    print(f"Finished gnina scoring for {outputs_dir_path}")
    return all_failed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run gnina scoring on model outputs.")

    # 2. Add arguments for the output path
    parser.add_argument("--exp_name", type=str, default="posebusters", help="Experiment name.")
    parser.add_argument("--model_id", type=str, default="model_0", help="Model ID.")
    parser.add_argument("--run_tag", type=str, help="Run tag.")

    # 3. Add arguments for the gnina config
    parser.add_argument("--preprocess", action="store_true", help="Enable preprocessing.")
    parser.add_argument("--no_gpu", action="store_true", help="Turn off GPU usage.")
    parser.add_argument("--scoring", type=str, default=None, help="Scoring function to use.")
    parser.add_argument("--device", type=int, default=0, help="GPU device to use.")

    # 4. Parse the arguments from the command line
    args = parser.parse_args()

    gnina_config = {
        "preprocess": args.preprocess,
        "no_gpu": args.no_gpu,
        "device": args.device,
    }
    scoring = args.scoring
    exp_name = args.exp_name
    model_id = args.model_id
    run_tag = args.run_tag

    print(f"args: {args}")

    outputs_dir_path = ROOT_DIR / "results" / exp_name / model_id / run_tag

    all_failed = gnina_score_for_outputs(outputs_dir_path, scoring=scoring, **gnina_config)
    print("All failed:", all_failed)
