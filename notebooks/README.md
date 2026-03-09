# SigmaDock notebooks

Run from the **project root** so paths and imports resolve correctly.

## Main notebooks

| Notebook | Description |
|----------|-------------|
| **01_visualize_data_and_pocket.ipynb** | Load a complex, inspect pocket definition (distance cutoff, CoM) and optionally view 3D (py3Dmol/plotly). |
| **02_alignment.ipynb** | Conformer alignment (Kabsch) and comparison: `ConformerOptimizer`, `visualize_comparison`, pairplots. |
| **03_fragmentation.ipynb** | **FRED**: Fragment a molecule at torsional bonds, compare strategies, enumerate valid fragmentations, and visualize fragments (grid + on molecule). |
| **04_forward_diffusion.ipynb** | **Forward diffusion**: Noise schedules, forward marginal, data + trajectory; batch multi-seed sampling, PoseBusters, and top-k heuristics. For a more detailed sampling pipeline see **extensions/sampling.ipynb**. |
| **05_parsing_and_processing.ipynb** | **sample.py outputs**: Load `predictions.pt` (and optionally `rescoring.pt`, `posebusters.pt`). See 04 and **extensions/sampling.ipynb** for full analysis; **examples/statistics.ipynb** for Vina scoring. |
| **05_crossdock_sampling.ipynb** | **Cross-docking** on **dummy_data**: DataFront from `dummy_crossdock` config (`query_*.sdf` = ligands to dock, `*_ligand.sdf` = reference). Run `dummy_data/setup_crossdock_queries.py` first. |

## Extensions

| Notebook | Description |
|----------|-------------|
| **extensions/sampling.ipynb** | Detailed extension of 04: full large-batch multi-seed sampling, PoseBusters checks, and top-k heuristics. More seeds, config options, and analysis than the compact version in 04. |

## Environment (optional)

- `SIGMADOCK_DATA_DIR`: path to data root (default: `data/`).
- `SIGMADOCK_CKPT_DIR`: path to checkpoint for the sampling notebook (default: `checkpoints/last.ckpt`).

**How to set them:**

1. **Shell (before starting Jupyter):**
   ```bash
   export SIGMADOCK_DATA_DIR=/path/to/your/data
   export SIGMADOCK_CKPT_DIR=/path/to/model.ckpt
   jupyter notebook   # or: jupyter lab
   ```

2. **Inside a notebook (first cell):**
   ```python
   import os
   os.environ["SIGMADOCK_DATA_DIR"] = "/path/to/your/data"
   os.environ["SIGMADOCK_CKPT_DIR"] = "/path/to/model.ckpt"
   ```
   Run that cell before any cell that uses `DATA_DIR` or `CKPT_PATH`.

3. **`.env` file** in the project root: if you use `python-dotenv` you can `load_dotenv()` in the first cell; the notebooks do not load `.env` by default.

**Install notebook deps:** `pip install -e ".[dev,test]"` (adds jupyter, py3Dmol, plotly, spyrmsd).

**Conformer viz:** Shared helpers for aligning and viewing multiple conformers live in `sigmadock.chem.conformer_viz` (e.g. `view_aligned_conformers`, `align_conformers`).
