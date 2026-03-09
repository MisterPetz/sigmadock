#!/usr/bin/env python3
"""
Populate cross-docking query ligands for dummy_data.

For each subfolder (PDB_LIG), copies the reference ligand SDF from every *other*
subfolder into it as query_<SOURCE_PDB>_<SOURCE_LIG>.sdf. Use regex `query_.*\\.sdf`
for "ligands to dock" and `.*_ligand\\.sdf` (or your protein's native ligand file)
as reference when running cross-docking.
"""
from pathlib import Path

DUMMY_DATA = Path(__file__).resolve().parent
REF_LIGAND_SUFFIX = "_ligand.sdf"


def main() -> None:
    subdirs = sorted([p for p in DUMMY_DATA.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not subdirs:
        print("No subfolders found.")
        return

    for target_dir in subdirs:
        target_id = target_dir.name  # e.g. 1G9V_RQ3
        ref_ligand = target_dir / f"{target_id}{REF_LIGAND_SUFFIX}"
        if not ref_ligand.exists():
            print(f"Skip {target_id}: no {ref_ligand.name}")
            continue
        for source_dir in subdirs:
            if source_dir == target_dir:
                continue
            source_id = source_dir.name
            source_ligand = source_dir / f"{source_id}{REF_LIGAND_SUFFIX}"
            if not source_ligand.exists():
                print(f"Skip {source_id}: no {source_ligand.name}")
                continue
            out_path = target_dir / f"query_{source_id}.sdf"
            out_path.write_bytes(source_ligand.read_bytes())
            print(f"  {target_id}/query_{source_id}.sdf <- {source_id}/{source_ligand.name}")

    print(f"Done. {len(subdirs)} folders; query SDFs use regex: query_.*\\.sdf")


if __name__ == "__main__":
    main()
