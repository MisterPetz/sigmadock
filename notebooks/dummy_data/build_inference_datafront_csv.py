#!/usr/bin/env python3
"""Write inference_datafront.csv: one row per query_*.sdf in each complex folder."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "inference_datafront.csv"


def main() -> None:
    rows: list[tuple[str, str, str]] = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        pdb = d / f"{name}_protein.pdb"
        ref = d / f"{name}_ligand.sdf"
        if not pdb.is_file() or not ref.is_file():
            continue
        for q in sorted(d.glob("query_*.sdf")):
            rows.append((f"{name}/{pdb.name}", f"{name}/{q.name}", f"{name}/{ref.name}"))

    lines = ["PDB,SDF,REF_SDF", *(",".join(r) for r in rows)]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
