import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from sigmadock.config import ExperimentConfig, get_experiment_config


@dataclass
class DataFront:
    dataroot: Path
    pdb_regex: str
    sdf_regex: str
    ref_sdf_regex: Optional[str] = None
    pairs: list[tuple[str, str, Optional[str]]] = field(default_factory=list)  # (sdf_path, pdb_path, ref_sdf_path | None)  # noqa: E501

    def __post_init__(self) -> None:
        if not isinstance(self.dataroot, Path):
            self.dataroot = Path(self.dataroot)

        self._pdb_pattern = re.compile(self.pdb_regex)
        self._sdf_pattern = re.compile(self.sdf_regex)
        self._ref_sdf_pattern = re.compile(self.ref_sdf_regex) if self.ref_sdf_regex else None

        self._setup()

    def _find_ref_sdf_in_folder(self, folder: Path) -> Optional[str]:
        """Return relative path to first SDF in folder matching ref_sdf_regex, or None."""
        if self._ref_sdf_pattern is None:
            return None
        for f in folder.iterdir():
            if f.suffix.lower() == ".sdf" and self._ref_sdf_pattern.search(f.name):
                return str(f.relative_to(self.dataroot))
        return None

    def _setup(self) -> None:  # noqa: C901
        # Traverse top-level directories (assumed to be pdb_ids or similar)
        num_visible = 0
        for subdir in sorted(self.dataroot.iterdir()):
            if not subdir.is_dir():
                continue
            pdb_matches = []
            sdf_matches = []

            for file_path in subdir.iterdir():
                if file_path.suffix.lower() == ".pdb" and self._pdb_pattern.search(file_path.name):
                    pdb_matches.append(file_path.relative_to(self.dataroot))
                elif file_path.suffix.lower() == ".sdf" and self._sdf_pattern.search(file_path.name):
                    sdf_matches.append(file_path.relative_to(self.dataroot))

            if (len(pdb_matches) > 1) or ((len(sdf_matches) > 1) and self.ref_sdf_regex is None):
                print(
                    f"Warning: Found multiple matches in {subdir}. ",
                    f"Using all matches for {self.dataroot} with re {self.pdb_regex} and {self.sdf_regex}",
                )

            ref_sdf_rel = self._find_ref_sdf_in_folder(subdir)
            for sdf_file in sdf_matches:
                for pdb_file in pdb_matches:
                    self.pairs.append(
                        (
                            str(sdf_file),
                            str(pdb_file),
                            ref_sdf_rel,
                        )
                    )
            num_visible += len(pdb_matches) * len(sdf_matches)
        if len(self.pairs) == 0:
            raise ValueError(f"No valid pairs found in {self.dataroot}. Check your regex patterns.")
        if num_visible != len(self.pairs):
            print(
                f"Warning: {num_visible} directories found, but only {len(self.pairs)} valid pairs."
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Path, Path, Optional[Path]]:
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError("Index out of range")
        pair = self.pairs[idx]
        ref_path = (self.dataroot / pair[2]) if pair[2] is not None else None
        return (self.dataroot / pair[0], self.dataroot / pair[1], ref_path)

    def __repr__(self) -> str:
        return f"DataFront(dataroot={self.dataroot}, pairs={len(self.pairs)})"

    def load_esm_embeddings(self) -> tuple[dict[str, torch.Tensor], dict[str, dict[tuple[str, int, str], int]]]:
        all_embeddings_dict = torch.load(self.dataroot / "esm_embeddings.pt")
        index_dict = torch.load(self.dataroot / "esm_embeddings_idx.pt")
        return all_embeddings_dict, index_dict

    def filter_data_without_embeddings(self) -> None:
        original_len = len(self.pairs)
        pdb_names_without_embeddings = torch.load(self.dataroot / "pdb_names_without_embeddings.pt")
        filtered_pairs = []
        for sdf_rel, pdb_rel, ref_rel in self.pairs:
            pdb_path = self.dataroot / pdb_rel
            if pdb_path.parent.stem in pdb_names_without_embeddings:  # e.g. 1a30 for .../1a30_pocket.pdb
                continue
            filtered_pairs.append((sdf_rel, pdb_rel, ref_rel))
        self.pairs = filtered_pairs
        print(f"Filtered data without ESM3 embeddings: {len(self.pairs)}/{original_len}.")


class MetaFront:
    """
    A unified DataFront over multiple DataFronts or ExperimentConfigs.
    Automatically uses each child's `dataroot`.

    You can pass a list of:
      - DataFront instances
      - ExperimentConfig instances
      - Experiment-name strings (will be loaded via `get_experiment_config`)
    """

    def __init__(
        self,
        inputs: list[DataFront | ExperimentConfig | str],
    ) -> None:
        self.fronts: list[DataFront] = []
        self.pairs: list[tuple[Path, Path, Optional[Path]]] = []

        for inp in inputs:
            if isinstance(inp, DataFront):
                df = inp
            elif isinstance(inp, ExperimentConfig):
                df = DataFront(
                    inp.dataset,
                    inp.pdb_regex,
                    inp.sdf_regex,
                    ref_sdf_regex=getattr(inp, "ref_sdf_regex", None),
                )
            elif isinstance(inp, str):
                cfg = get_experiment_config(inp)
                df = DataFront(
                    cfg.dataset,
                    cfg.pdb_regex,
                    cfg.sdf_regex,
                    ref_sdf_regex=getattr(cfg, "ref_sdf_regex", None),
                )
            else:
                raise TypeError(f"Unsupported input type: {type(inp)}")

            self.fronts.append(df)

            for sdf_rel, pdb_rel, ref_rel in df.pairs:
                ref_path = (df.dataroot / ref_rel) if ref_rel else None
                self.pairs.append((df.dataroot / sdf_rel, df.dataroot / pdb_rel, ref_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Path, Path, Optional[Path]]:
        if idx < 0 or idx >= len(self.pairs):
            raise IndexError("Index out of range")
        return self.pairs[idx]

    def __repr__(self) -> str:
        roots = [str(df.dataroot) for df in self.fronts]
        return f"MetaFront(roots={roots}, total_pairs={len(self)})"

    def load_esm_embeddings(self) -> tuple[dict[str, torch.Tensor], dict[str, dict[tuple[str, int, str], int]]]:
        all_embeddings_dict_list = []
        index_dict_list = []
        for df in self.fronts:
            all_embeddings_dict_df = torch.load(df.dataroot / "esm_embeddings.pt")
            index_dict_df = torch.load(df.dataroot / "esm_embeddings_idx.pt")
            all_embeddings_dict_list.append(all_embeddings_dict_df)
            index_dict_list.append(index_dict_df)
        all_embeddings_dict = {k: v for d in all_embeddings_dict_list for k, v in d.items()}
        index_dict = {k: v for d in index_dict_list for k, v in d.items()}
        return all_embeddings_dict, index_dict

    def filter_data_without_embeddings(self) -> None:
        original_len = len(self.pairs)
        all_pdb_names_without_embeddings = []
        for df in self.fronts:
            pdb_names_without_embeddings = torch.load(df.dataroot / "pdb_names_without_embeddings.pt")
            all_pdb_names_without_embeddings += pdb_names_without_embeddings
        filtered_pairs = []
        for sdf_path, pdb_path, ref_path in self.pairs:
            if pdb_path.parent.stem in all_pdb_names_without_embeddings:  # e.g. 1a30 for .../1a30_pocket.pdb
                continue
            filtered_pairs.append((sdf_path, pdb_path, ref_path))
        self.pairs = filtered_pairs
        print(f"Filtered data without ESM3 embeddings: {len(self.pairs)}/{original_len}.")

    def prune_pairs_with_ids(self, ids: list[str]) -> None:
        """
        Prune pairs to only include those with PDB IDs in the provided list.
        """
        original_len = len(self.pairs)
        filtered_pairs = []
        for sdf_path, pdb_path, ref_path in self.pairs:
            pdb_id = pdb_path.parent.stem
            if pdb_id in ids:
                filtered_pairs.append((sdf_path, pdb_path, ref_path))
        self.pairs = filtered_pairs
        print(f"Pruned pairs to {len(self.pairs)}/{original_len} using provided IDs.")

def get_datafront(
    dataset_path: str | Path,
    pdb_regex: str | None = None,
    sdf_regex: str | None = None,
    ref_sdf_regex: Optional[str] = None,
) -> DataFront | None:
    """
    Given a dataset path, derive the DataFront object.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists() or dataset_path.stem == "":
        return None
    if pdb_regex is None:
        pdb_regex = r""
    if sdf_regex is None:
        sdf_regex = r".*ligands.*\.sdf$"

    try:
        re.compile(pdb_regex)
        re.compile(sdf_regex)
        if ref_sdf_regex is not None:
            re.compile(ref_sdf_regex)
    except re.error as e:
        raise ValueError(f"Invalid regex: {e}") from e

    datafront = DataFront(
        dataset_path,
        pdb_regex=pdb_regex,
        sdf_regex=sdf_regex,
        ref_sdf_regex=ref_sdf_regex,
    )
    assert datafront is not None, f"DataFront could not be created for {dataset_path}"
    if len(datafront) == 0:
        print(f"DataFront is empty for {dataset_path}")
        return None

    print(f"DataFront created for {dataset_path} with {len(datafront)} samples.")
    return datafront
