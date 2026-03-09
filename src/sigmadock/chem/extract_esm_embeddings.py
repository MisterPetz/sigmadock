import contextlib
from pathlib import Path
from typing import Optional

import biotite.structure as bs
import numpy as np
import torch
from Bio import PDB
from Bio.Data.PDBData import protein_letters_3to1
from biotite.structure.io.pdb import PDBFile
from esm.models.esm3 import ESM3  # type: ignore
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig  # type: ignore
from esm.utils.structure.protein_complex import ProteinComplex  # type: ignore
from rdkit import Chem
from rdkit.Chem.rdchem import Atom

from sigmadock.chem.parsing import (
    get_protein,
    read_ligands_from_sdf,
)
from sigmadock.chem.processing import get_coordinates
from sigmadock.datafronts import DataFront

SPECIAL_TOKENS = [0, 1, 2, 3, 31, 32]  # 31 declares the start of a new chain


def get_esm_embedding_for_atom(
    esm_embeddings_pdb: torch.Tensor, esm_embeddings_pdb_idx: dict[tuple[str, int, str], int], atom: Atom
) -> torch.Tensor:
    info = atom.GetPDBResidueInfo()
    chain_id = info.GetChainId() or " "
    residue_number = info.GetResidueNumber()
    insertion_code = info.GetInsertionCode() or " "

    key = (chain_id, residue_number, insertion_code)
    embedding_idx = esm_embeddings_pdb_idx.get(key)
    if embedding_idx is None:
        raise KeyError(f"Residue {key} not found in cached ESM embeddings")
    return esm_embeddings_pdb[embedding_idx]


def get_esm_residue_sequence(pdb_path: Path) -> list[tuple[str, tuple[str, int, str]]]:
    """Extracts the residue sequence from a PDB file following ESM conventions."""

    atom_array = PDBFile.read(pdb_path).get_structure(model=1, extra_fields=["b_factor"])
    residue_sequence: list[tuple[str, tuple[str, int, str]]] = []

    for chain in bs.chain_iter(atom_array):
        # Filter out hetero atoms and non-amino acids
        atom_array_for_chain = chain[bs.filter_amino_acids(chain) & ~chain.hetero]
        # Replace non-standard amino acids with 'X'
        # We use " " for no chain to align with PDBParser conventions
        subsequence = [
            (
                (r, (monomer[0].chain_id.item() or " ", monomer[0].res_id.item(), monomer[0].ins_code.item() or " "))
                if len(r := protein_letters_3to1.get(monomer[0].res_name, "X")) == 1
                else "X"
            )
            for monomer in bs.residue_iter(atom_array_for_chain)
        ]
        residue_sequence += subsequence

    return residue_sequence


def process_pdb_for_embeddings(
    pdb_path: Path, model: ESM3InferenceClient, max_seq_length: int | None = None
) -> dict[tuple[str, int, str], torch.Tensor] | None:
    """Processes a PDB file to extract ESM3 embeddings for each residue."""

    # Load tokeniser vocabulary
    vocab = model.tokenizers.sequence.vocab
    vocab = {v: k for k, v in vocab.items()}

    # Convert PDB file
    try:
        protein_complex = ProteinComplex.from_pdb(pdb_path)
        protein = ESMProtein.from_protein_complex(protein_complex)
        print(f"ESMProtein length: {len(protein.sequence)}.")
    except Exception as e:
        print(f"[WARNING] Failed to process {pdb_path.name} as ESMProtein. Skipping.")
        return None

    # Filter file by sequence length
    if max_seq_length is not None and len(protein.sequence) > max_seq_length:
        print(f"Skipping {pdb_path.name} due to length {len(protein.sequence)} > {max_seq_length}.")
        return None

    # Compute ESM3 embeddings
    with torch.no_grad():
        protein_tensor = model.encode(protein)
        mask = torch.isin(protein_tensor.sequence, torch.tensor(SPECIAL_TOKENS, device=model.device))
        output = model.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))

    embeddings = output.per_residue_embedding[~mask].detach().to("cpu")  # [:, 1536]
    sequence = protein_tensor.sequence[~mask].detach().to("cpu")

    # per_residue_embedding returns the pre-layer norm embeddings, so we apply the layer norm
    # for better conditioned embeddings.
    with (
        torch.no_grad(),  # Assume no gradients for now...
        torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16)  # type: ignore
        if device.type == "cuda"
        else contextlib.nullcontext(),
    ):
        embeddings = model.transformer.norm(embeddings)

    assert len(embeddings) == len(sequence), "Mismatch between embeddings and sequence length."
    print(f"Extracted {embeddings.shape[0]} embeddings.")

    # Associate embeddings with residue information
    residue_sequence = get_esm_residue_sequence(pdb_path)
    assert len(residue_sequence) == len(embeddings), (
        f"Mismatch between residue sequence {len(residue_sequence)} and embeddings length {len(embeddings)}."
    )

    emb_dict: dict[tuple[str, int, str], torch.Tensor] = {}

    for idx, (res_name, res_info) in enumerate(residue_sequence):
        esm_residue_name = vocab[sequence[idx].item()]
        assert esm_residue_name == res_name, (
            f"Mismatch between ESM residue name {esm_residue_name} and PDB residue name {res_name} at idx={idx}."
        )
        emb_dict[res_info] = embeddings[idx]

    print(f"{len(emb_dict)} residues found in {pdb_path.name}.")

    return emb_dict


def extract_pocket_embeddings_from_ligand(
    emb_dict: dict[tuple[str, int, str], torch.Tensor], lig_path: Path, pdb_path: Path, distance_cutoff: float
) -> dict[tuple[str, int, str], torch.Tensor] | None:
    """Filter emb_dict to only include residues within distance_cutoff of a ligand atom."""

    # Load ligands
    try:
        ligand_mols: list[Chem.Mol] = read_ligands_from_sdf(lig_path)
        if not ligand_mols:
            raise ValueError(f"No valid ligands found in SDF file: {lig_path}")
    except Exception as e:
        print(f"[WARNING] Failed to read ligands from {lig_path}. Skipping.")
        return None

    # Load protein
    structure = get_protein(pdb_path)

    # Extract residues within the distance cutoff from the ligand
    pocket_residues: set[tuple[str, int, str]] = set()

    for ligand_mol in ligand_mols:
        ligand_coords = get_coordinates(ligand_mol, heavy_only=True)
        if ligand_coords.size == 0:
            continue  # Skip ligands with no valid coordinates

        for residue in structure.get_residues():
            # We only consider standard amino acids
            if not PDB.is_aa(residue, standard=True):
                continue
            chain_id = residue.get_parent().id or " "  # use space for “no chain”
            resseq = residue.id[1]
            icode = residue.id[2] or " "  # use space for “no insertion”

            res_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
            if res_coords.size == 0:
                continue

            # Compute distances to ligand atoms
            distances = np.linalg.norm(res_coords[:, None, :] - ligand_coords[None, :, :], axis=-1)
            if np.any(distances < distance_cutoff):
                pocket_residues.add((chain_id, resseq, icode))

    # Keep only embeddings for residues within range of the distance cutoff
    original_residues_count = len(emb_dict)
    emb_dict = {k: emb_dict[k] for k in pocket_residues}

    print(f"Kept {len(emb_dict)}/{original_residues_count} residues in the pocket for {pdb_path.name}.")
    return emb_dict


def cache_esm_embeddings_from_datafront(
    datafront: DataFront,
    distance_cutoff: float,
    max_seq_length: Optional[int] = 3000,
    device: str = "cpu",
) -> list[str]:
    """Saves ESM3 embeddings for each PDB file in datafront to a temporary directory."""

    print(f"Extracting ESM3 embeddings for PDB files in {datafront.dataroot}...")

    if max_seq_length is not None:
        print(f"Max sequence length set to {max_seq_length}.")

    # Load model once
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open", device=torch.device(device))
    model.eval()
    print(f"Using device: {device} for ESM3 model.")

    pdb_names_without_embeddings = []
    tmp_path = datafront.dataroot / "tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)

    for lig_path, pdb_path, _ in datafront:
        print(f"\nProcessing {pdb_path.name}...")

        # Compute ESM3 embeddings for the whole protein
        emb_dict = process_pdb_for_embeddings(pdb_path, model, max_seq_length=max_seq_length)
        if emb_dict is None:
            pdb_names_without_embeddings.append(pdb_path.parent.stem)  # e.g. 1a30 for the PDB file 1a30_protein.pdb
            continue

        # Filter embeddings by residues within the distance cutoff from the ligand
        emb_dict = extract_pocket_embeddings_from_ligand(emb_dict, lig_path, pdb_path, distance_cutoff)
        if emb_dict is None or len(emb_dict) == 0:
            pdb_names_without_embeddings.append(pdb_path.parent.stem)
            continue

        # Save hyperparameters
        emb_dict["distance_cutoff"] = distance_cutoff
        emb_dict["max_seq_length"] = max_seq_length

        file_name = pdb_path.parent.stem + ".pt"  # e.g. 1a30.pt for the PDB file 1a30_protein.pdb
        torch.save(emb_dict, tmp_path / file_name)

        # Prevent memory leakage
        del emb_dict
        torch.cuda.empty_cache()

    torch.save(pdb_names_without_embeddings, datafront.dataroot / "pdb_names_without_embeddings.pt")

    print(f"Finished extracting ESM3 embeddings for all PDB files in {datafront.dataroot}.")
    return pdb_names_without_embeddings


def collect_cache_esm_embeddings(
    datafront: DataFront,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[tuple[str, int, str], int]]]:
    """Aggregates ESM3 embeddings from the temporary directory into a dictionaries
    for embeddings and residue indexes."""

    tmp_path = datafront.dataroot / "tmp"
    pdb_names_without_embeddings = torch.load(datafront.dataroot / "pdb_names_without_embeddings.pt")

    all_embeddings_dict: dict[str, torch.Tensor] = {}
    index_dict: dict[str, dict[tuple[str, int, str], int]] = {}

    distance_cutoff = None
    max_seq_length = None

    for pt_path in tmp_path.glob("*.pt"):  # e.g. .../1a30.pt for the PDB file 1a30_protein.pdb
        print(f"\nProcessing {pt_path.name}...")
        emb_dict = torch.load(pt_path)
        _distance_cutoff = emb_dict.pop("distance_cutoff", None)
        _max_seq_length = emb_dict.pop("max_seq_length", None)

        if len(emb_dict) == 0:
            print(f"No valid embeddings found in {pt_path.name}. Skipping.")
            pdb_names_without_embeddings.append(pt_path.stem)
            continue

        embeddings_tensor = torch.stack(list(emb_dict.values()), dim=0)
        all_embeddings_dict[pt_path.stem] = embeddings_tensor  # index by 1a30 for the PDB file 1a30_protein.pdb
        index_dict[pt_path.stem] = {k: i for i, k in enumerate(emb_dict.keys())}

        # Check if the distance cutoff and max sequence length are consistent
        if distance_cutoff is None:
            distance_cutoff = _distance_cutoff
        elif distance_cutoff != _distance_cutoff:
            raise ValueError(f"Distance cutoff mismatch: {distance_cutoff} != {_distance_cutoff}")
        if max_seq_length is None:
            max_seq_length = _max_seq_length
        elif max_seq_length != _max_seq_length:
            raise ValueError(f"Max sequence length mismatch: {max_seq_length} != {_max_seq_length}")

    index_dict["distance_cutoff"] = distance_cutoff
    index_dict["max_seq_length"] = max_seq_length

    torch.save(all_embeddings_dict, datafront.dataroot / "esm_embeddings.pt")
    torch.save(index_dict, datafront.dataroot / "esm_embeddings_idx.pt")
    torch.save(pdb_names_without_embeddings, datafront.dataroot / "pdb_names_without_embeddings.pt")

    print(f"Finished collecting ESM3 embeddings for all PDB files in {datafront.dataroot}.")
    return all_embeddings_dict, index_dict


if __name__ == "__main__":
    # pass

    # import os
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # handles memory issues with large proteins

    CORE_DATAROOT = Path("/data/ziz/not-backed-up/datasets-ziz-all/sigmadock/data/pdbbind/core-set/")
    GENERAL_DATAROOT = Path("/data/ziz/not-backed-up/datasets-ziz-all/sigmadock/data/pdbbind/general-set/")
    REFINED_DATAROOT = Path("/data/ziz/not-backed-up/datasets-ziz-all/sigmadock/data/pdbbind/refined-set/")

    POSEBUSTERS_DATAROOT = Path(
        "/data/ziz/not-backed-up/datasets-ziz-all/sigmadock/data/posebusters_paper/posebusters_benchmark_set/"
    )
    ASTEX_DATAROOT = Path(
        "/data/ziz/not-backed-up/datasets-ziz-all/sigmadock/data/posebusters_paper/astex_diverse_set/"
    )

    PDB_DATAROOTS = [CORE_DATAROOT, GENERAL_DATAROOT, REFINED_DATAROOT]
    POSEBUSTERS_DATAROOTS = [POSEBUSTERS_DATAROOT, ASTEX_DATAROOT]

    device = "cuda"

    distance_cutoff = 8.0
    max_seq_length = 3000

    for dataroot in PDB_DATAROOTS:
        datafront = DataFront(dataroot, pdb_regex=".*protein\\.pdb$", sdf_regex=".*ligand.*\\.sdf$")

        # _ = cache_esm_embeddings_from_datafront(
        #     datafront=datafront,
        #     distance_cutoff=distance_cutoff,
        #     max_seq_length=max_seq_length,
        #     device=device,
        # )
        _ = collect_cache_esm_embeddings(datafront)

    for dataroot in POSEBUSTERS_DATAROOTS:
        datafront = DataFront(dataroot, pdb_regex=".*\\.pdb$", sdf_regex=".*ligands.*\\.sdf$")

        # _ = cache_esm_embeddings_from_datafront(
        #     datafront=datafront,
        #     distance_cutoff=distance_cutoff,
        #     max_seq_length=max_seq_length,
        #     device=device,
        # )
        _ = collect_cache_esm_embeddings(datafront)
