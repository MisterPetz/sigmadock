"""Hydra config, data-directory resolution, and datafront construction for ``scripts/sample.py``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from omegaconf import MISSING, DictConfig, OmegaConf

from sigmadock.config import get_experiment_config
from sigmadock.datafronts import ExplicitPairsFront, MetaFront


def merge_legacy_sampling_subconfig(hcfg: DictConfig) -> DictConfig:
    """If present, merge ``sampling.*`` onto the root config (legacy CLI)."""
    if "sampling" not in hcfg or OmegaConf.is_missing(hcfg, "sampling"):
        return hcfg
    sub = hcfg.sampling
    if sub is None:
        return hcfg
    try:
        sub_d = OmegaConf.to_container(sub, resolve=True)
    except Exception:
        return hcfg
    if not isinstance(sub_d, dict) or not sub_d:
        return hcfg
    root_d = OmegaConf.to_container(hcfg, resolve=True)
    if not isinstance(root_d, dict):
        return hcfg
    root_d.pop("sampling", None)
    base = OmegaConf.create(root_d)
    return OmegaConf.merge(base, sub)


def apply_sampling_cli_aliases(cfg: DictConfig) -> None:
    """Map top-level ``ckpt``, ``data_dir``, ``experiment`` into nested fields."""
    OmegaConf.set_struct(cfg, False)
    if nonempty_cfg_str(OmegaConf.select(cfg, "ckpt")):
        cfg.model.ckpt_dir = OmegaConf.select(cfg, "ckpt")
    if nonempty_cfg_str(OmegaConf.select(cfg, "data_dir")):
        cfg.data.data_dir = OmegaConf.select(cfg, "data_dir")
    if nonempty_cfg_str(OmegaConf.select(cfg, "experiment")):
        if "experiments" not in cfg:
            cfg.experiments = OmegaConf.create({"name": None})
        cfg.experiments.name = OmegaConf.select(cfg, "experiment")


def prepare_sampling_cfg(hcfg: DictConfig) -> DictConfig:
    """Legacy merge + CLI aliases; returns the config object to use (often same as ``hcfg``)."""
    cfg = merge_legacy_sampling_subconfig(hcfg)
    apply_sampling_cli_aliases(cfg)
    return cfg


def nonempty_cfg_str(x: Any) -> bool:
    """True for a non-empty string leaf (``OmegaConf.select`` / resolved values, not config nodes)."""
    if x is None or x is MISSING:
        return False
    return str(x).strip() != ""


def experiment_name_is_set(name: Any) -> bool:
    if name is None or name is MISSING:
        return False
    s = str(name).strip().lower()
    return s not in ("", "none", "null", "~")


def _resolve_csv_cell(cell: str, base_dir: Path) -> Path:
    raw = cell.strip().strip('"').strip("'")
    if not raw:
        raise ValueError("Empty path cell in inference_datafront CSV")
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (base_dir / p).resolve()


def _datafront_csv_headers(fieldnames: list[str] | None) -> tuple[str, str, str | None]:
    if not fieldnames:
        raise ValueError("inference_datafront CSV has no header row")
    upper: dict[str, str] = {}
    for raw in fieldnames:
        if raw is None:
            continue
        upper[raw.strip().upper().replace(" ", "_")] = raw.strip()
    if "PDB" not in upper or "SDF" not in upper:
        raise ValueError(
            "inference_datafront CSV must include PDB and SDF columns; "
            f"optional REF_SDF (REFERENCE_SDF, REF). Headers: {fieldnames!r}"
        )
    ref_col = None
    for key in ("REF_SDF", "REFERENCE_SDF", "REF"):
        if key in upper:
            ref_col = upper[key]
            break
    return upper["PDB"], upper["SDF"], ref_col


def load_inference_datafront_csv(csv_path: Path) -> list[tuple[str, str, str | None]]:
    """Rows as ``(ligand_sdf, protein_pdb, reference_sdf?)``; relative paths use ``csv_path``'s parent."""
    base = csv_path.parent.resolve()
    pairs: list[tuple[str, str, str | None]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        pdb_h, sdf_h, ref_h = _datafront_csv_headers(reader.fieldnames)
        for i, row in enumerate(reader, start=2):
            if row is None:
                continue
            pc = (row.get(pdb_h) or "").strip()
            sc = (row.get(sdf_h) or "").strip()
            if not pc and not sc:
                continue
            if not pc or not sc:
                raise ValueError(f"inference_datafront CSV row {i}: PDB and SDF must both be set")
            ref: str | None = None
            if ref_h:
                rc = (row.get(ref_h) or "").strip()
                if rc:
                    ref = str(_resolve_csv_cell(rc, base))
            pairs.append((str(_resolve_csv_cell(sc, base)), str(_resolve_csv_cell(pc, base)), ref))
    if not pairs:
        raise ValueError(f"No data rows in inference_datafront CSV: {csv_path}")
    return pairs


def build_sampling_datafront(cfg: DictConfig, data_dir: Path) -> MetaFront | ExplicitPairsFront:
    inf = cfg.get("inference") or OmegaConf.create({})

    if experiment_name_is_set(cfg.experiments.get("name")):
        ec = get_experiment_config(str(cfg.experiments.name), data_dir)
        if "sdf_regex" in cfg.experiments:
            ec.sdf_regex = str(cfg.experiments.sdf_regex)
        if "pdb_regex" in cfg.experiments:
            ec.pdb_regex = str(cfg.experiments.pdb_regex)
        if getattr(cfg.experiments, "ref_sdf_regex", None) is not None:
            ec.ref_sdf_regex = str(cfg.experiments.ref_sdf_regex)
        df = MetaFront([ec])
        if getattr(ec, "ref_sdf_regex", None) is not None:
            print(f"Using ref_sdf_regex for <Cross-Docking> pocket definition: {ec.ref_sdf_regex}")
        else:
            print("No reference SDF found, using <Re-Docking> pocket definition.")
        return df

    idf = OmegaConf.select(inf, "inference_datafront", default=None)
    if nonempty_cfg_str(idf):
        p = Path(str(idf)).expanduser().resolve()
        if not p.is_file():
            raise ValueError(f"inference.inference_datafront is not a file: {p}")
        pairs = load_inference_datafront_csv(p)
        print(f"Using inference_datafront ({p}): {len(pairs)} complex(es).")
        return ExplicitPairsFront(pairs)

    lig = OmegaConf.select(inf, "ligand_sdf", default=None)
    pdb = OmegaConf.select(inf, "protein_pdb", default=None)
    if nonempty_cfg_str(lig) and nonempty_cfg_str(pdb):
        ref = OmegaConf.select(inf, "reference_sdf", default=None)
        ref_path = str(ref).strip() if nonempty_cfg_str(ref) else None
        print("Using explicit inference paths (ligand_sdf, protein_pdb, reference_sdf?).")
        return ExplicitPairsFront([(str(lig), str(pdb), ref_path)])

    raise ValueError(
        "Custom inference: set inference.inference_datafront=... (CSV with PDB, SDF), or "
        "inference.ligand_sdf + inference.protein_pdb (+ optional reference_sdf), "
        "or experiment=... for a benchmark under conf/experiments/."
    )


def sampling_results_exp_name(cfg: DictConfig) -> str:
    if experiment_name_is_set(cfg.experiments.get("name")):
        return str(cfg.experiments.name)
    if nonempty_cfg_str(OmegaConf.select(cfg, "run_tag", default=None)):
        return str(OmegaConf.select(cfg, "run_tag")).strip()
    # Legacy: inference.run_label before run_tag was used for custom runs
    inf = cfg.get("inference") or OmegaConf.create({})
    if nonempty_cfg_str(OmegaConf.select(inf, "run_label", default=None)):
        return str(OmegaConf.select(inf, "run_label")).strip()
    return "sampling"


def resolve_sampling_data_dir(cfg: DictConfig) -> Path:
    """Directory used as ``data_dir`` for benchmark layouts and existence checks."""
    data_dir_s = OmegaConf.select(cfg, "data.data_dir", default=None)
    inf = cfg.get("inference") or OmegaConf.create({})

    if experiment_name_is_set(cfg.experiments.get("name")):
        if not nonempty_cfg_str(data_dir_s):
            raise ValueError(
                "data.data_dir (or data_dir=...) is required when experiment=... selects a benchmark layout."
            )
        return Path(str(data_dir_s)).expanduser().resolve()

    lig = OmegaConf.select(inf, "ligand_sdf", default=None)
    pdb = OmegaConf.select(inf, "protein_pdb", default=None)
    idf = OmegaConf.select(inf, "inference_datafront", default=None)

    if nonempty_cfg_str(lig) and nonempty_cfg_str(pdb):
        if nonempty_cfg_str(data_dir_s):
            return Path(str(data_dir_s)).expanduser().resolve()
        return Path(str(pdb)).expanduser().resolve().parent

    if nonempty_cfg_str(idf):
        return Path(str(idf)).expanduser().resolve().parent

    if not nonempty_cfg_str(data_dir_s):
        raise ValueError(
            "data.data_dir (or data_dir=...) is required unless you set "
            "inference.inference_datafront or both inference.ligand_sdf and inference.protein_pdb."
        )
    return Path(str(data_dir_s)).expanduser().resolve()
