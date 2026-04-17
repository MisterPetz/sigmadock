# SigmaDock :fire:

Official implementation of:

> **SigmaDock: Untwisting Molecular Docking with Fragment-Based SE(3) Diffusion**  
> Alvaro Prat, Leo Zhang, Charlotte Deane, Yee Whye Teh, Garrett Morris  
> *International Conference on Learning Representations (ICLR), 2026*

This repository supports **training**, **sampling**, and **evaluation** for the ICLR submission.

**Contact:** [alvaro.prat@stats.ox.ac.uk](mailto:alvaro.prat@stats.ox.ac.uk) · or open an issue. We will reply as soon as possible!

**⚠️ Please Note** this is a beta release. APIs and behaviour may change in future versions. Stay tuned!

---

## Table of contents

- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Sampling](#sampling)
- [SLURM](#slurm)
- [Notebooks](#notebooks)
- [Quick reference](#quick-reference)
- [Citation](#citation)

---

## Installation

### Requirements

- **Python** ≥3.9, <3.13  
- **CUDA** (for GPU training/sampling)

### From source (recommended)

```bash
git clone https://github.com/alvaroprat97/sigmadock.git
cd sigmadock

conda create -y -n sigmadock python=3.12
conda activate sigmadock

bash install.sh
```

Specify your own cuda version if necessary (i.e. cu121):

```bash
bash install.sh cu121
```

Or also specify which extras you want (i.e train and test only):

```bash
bash install.sh cu126 train,test
```

**Extra dependencies:**


| Extra    | Use case  | Adds                                              |
| -------- | --------- | ------------------------------------------------- |
| *(none)* | Core only | Minimal deps                                      |
| `train`  | Training  | `wandb`, `hydra-core`, `omegaconf`, `posebusters` |
| `dev`    | Notebooks | `jupyterlab`, `ipykernel`, `py3Dmol`, etc.        |
| `test`   | Tests     | `pytest`, `spyrmsd`, etc.                         |


After install, from the project root:

```bash
python scripts/train.py --help
python scripts/sample.py --help
```

Or use console entry points (if installed with the scripts package):

```bash
training --help
sampling --help
```

---

## Data

Place all benchmark data under a single **data root** (e.g. `data/`). 

### Directory layout

Each experiment uses a **subdirectory** of the data root. Inside that subdirectory you must have **one folder per complex**. Each complex folder must contain:


| File type            | Required | Description                                                                                                                                                     |
| -------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Protein**          | Yes      | One `.pdb` file matching the experiment’s `pdb_regex` (e.g. pocket or full structure).                                                                          |
| **Ligand(s)**        | Yes      | One or more ligand files (`*.sdf`) matching the experiment’s `sdf_regex`. If an SDF fails validation, the loader tries a same-stem `*.mol2` in the same folder. |
| **Reference ligand** | Optional | One `.sdf` matching `ref_sdf_regex` (if set): pocket / CoM when it differs from the query ligand (e.g. **cross-docking**). Similar in spirit to **--autobox**.  |


**Example — re-docking (one ligand per complex):**

```
<data_root>/
  <experiment_subdir>/
    1abc/
      protein.pdb
      ligand.sdf
```

**Example — cross-docking (reference + query ligands):**

```
<data_root>/
  <experiment_subdir>/
    1abc/
      protein.pdb
      1abc_ligand.sdf          # reference (pocket/CoM); set ref_sdf_regex to match this
      query_2def.sdf           # ligands to dock; set sdf_regex to match these
      query_3ghi.sdf
```

Experiment subdirs and regexes are defined in `conf/experiments/*.yaml`. Key options:

- `**pdb_regex**` — pattern for the protein PDB (e.g. `.*pocket\.pdb$`).
- `**sdf_regex**` — pattern for the **ligand file(s) to dock** (e.g. `.*ligand.*\.sdf$` or `query_.*\.sdf$` for cross-docking).
- `**ref_sdf_regex`** — *(optional)* reference ligand SDF for pocket definition and CoM only. Omit for re-docking; set for cross-docking.

### PDBBind

1. **Download** from [PDBBind](https://www.pdbbind-plus.org.cn/download) (e.g. refined set, general set).
2. **Process**: Extract protein (pocket) PDB and ligand SDF per complex. Many pipelines give one folder per PDB ID with e.g. `*pocket.pdb` and `*ligand*.sdf`.
3. **Place** under the data root so paths match the experiment configs:
  - Refined: `<data_root>/pdbbind/refined-set/<pdb_id>/...`
  - General: `<data_root>/pdbbind/general-set/...`
  - Core (validation): `<data_root>/pdbbind/core-set/...`

Configs in `conf/experiments/` use `pdb_regex: ".*pocket\\.pdb$"` and `sdf_regex: ".*ligand.*\\.sdf$"`; adjust if your filenames differ.

### PoseBusters benchmark

1. **Download** the [PoseBusters benchmark](https://github.com/maabuu/posebusters) (benchmark set and/or correct IDs list).
2. **Arrange** so each complex has a folder with a `.pdb` and `ligands.sdf` (or whatever regex is used in `conf/experiments/posebusters.yaml`).
3. **Place** under the data root, e.g.:
  - `<data_root>/posebusters_paper/posebusters_benchmark_set/<id>/...`
  - Optional ID list: `<data_root>/posebusters_paper/posebusters_correct_ids.txt` (one PDB/system ID per line). With `experiment=posebusters`, pass `data.blacklist=<path>` to restrict to those IDs.

### Astex (PoseBusters-style)

Configured in `conf/experiments/astex.yaml`:

- Place data under: `<data_root>/posebusters_paper/astex_diverse_set/<id>/...`
- Same per-folder layout: `.pdb` and `ligands.sdf` (or as per `sdf_regex` / `pdb_regex`).

---

## Training

From the project root:

```bash
python scripts/train.py \
  --data_dir /path/to/data \
  --train_exps pdbbind-refined pdbbind-general \
  --val_exps pdbbind-core \
  --experiment my_run \
  --seed 0
```

**Important flags** (see `scripts/train.py --help`):


| Flag                                        | Description                                                                                     |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `--data_dir`                                | Path to the data root.                                                                          |
| `--train_exps`, `--val_exps`, `--test_exps` | Experiment names (must have matching `conf/experiments/<name>.yaml` and data under `data_dir`). |
| `--experiment`                              | Run name (logging and checkpoint subdirs).                                                      |
| `--resume_from_checkpoint`                  | Set to `true` or a checkpoint path to resume.                                                   |


Checkpoints and logs are written under the experiment directory (default: `exp_dir` in config). 

You can set different hyperparameters for the main training script. We recommend using the default in `/conf/training/slurm.yaml`

## Using the released checkpoint

A pretrained checkpoint is provided with the current GitHub release (see this repository's **Releases** page). After downloading it, you can run sampling as follows:

```bash
python scripts/sample.py ckpt=/path/to/downloaded_checkpoint.ckpt data_dir=/path/to/data experiment=posebusters
```

If you use the SLURM sampling script in `slurm/sample.sh`, set:

```bash
export CKPT_DIR=/path/to/downloaded_checkpoint.ckpt
sbatch slurm/sample.sh
```

---

## Sampling

Sampling uses **Hydra** with `conf/sampling/base.yaml`. From the project root:

```bash
python scripts/sample.py ckpt=/path/to/checkpoint.ckpt data_dir=/path/to/data experiment=posebusters data.batch_size=16
```

`experiment` selects `conf/experiments/<name>.yaml` (optional regex overrides under `experiments.*`).

YAML-only invocation:

```bash
python scripts/sample.py --config-name sampling/base --config-path conf/
```

### Custom inference (`experiments.name` null — the default)

**Note:** each query SDF path is expected to contain **a single ligand**. Files with multiple records are not expanded into separate runs. Prefer one ligand per file, or one CSV row per ligand SDF.

1. **Explicit files** — optional `data_dir`; if omitted, defaults to the protein PDB’s parent directory (used as the logical data root for Hydra paths):
  ```bash
   python scripts/sample.py \
     ckpt=/path/to/checkpoint.ckpt \
     inference.ligand_sdf=/path/to/query.sdf \
     inference.protein_pdb=/path/to/pocket.pdb \
     inference.reference_sdf=/path/to/reference.sdf
  ```
   Omit `inference.reference_sdf` for re-docking.
2. **CSV datafront** — set `inference.inference_datafront` to a CSV with columns **PDB** and **SDF** (required), and optionally **REF_SDF** (or **REFERENCE_SDF** / **REF**). Paths in the CSV may be absolute, or **relative to the folder that contains the CSV**. Example `inference_datafront.csv`:
  ```csv
   PDB,SDF,REF_SDF
   structures/1abc_pocket.pdb,structures/query_1.sdf,structures/1abc_ref.sdf
   structures/1abc_pocket.pdb,structures/query_2.sdf,structures/1abc_ref.sdf
  ```

### Outputs and multiple seeds

For **custom inference** (no `experiment=...`), sampling writes under `<output_dir>/results/<run_tag>/<model_id>/seed_<cfg.seed>/` and runs with defaults: 

- `output_dir` = project root
- `run_tag` = `sampling`
- `model_id` from the checkpoint filename unless `model.model_id` is set.

**Multiple draws (ranking / top-from-N):** for **full reproducibility**, keep `**num_seeds: 1`** and launch **separate runs** with different `seed` values. We recommend launching a **SLURM job array** (`slurm/sample.sh` wires `seed` to the array task). Each run writes its own `results/.../seed_<seed>/` tree. 

- **Rescoring** and **PoseBusters** run per job, and you can aggregate across directories with `sigmadock.chem.statistics`. Prefer job arrays over `num_seeds > 1` in a single process when you want independent, pinned samples (and to avoid an effective batch size of `batch_size × num_seeds`).

### Post-processing

- **ReScoring:** in `conf/sampling/base.yaml`, set `postprocessing.scoring` to `"vina"` or `"vinardo"` to use **GNINA** for rescoring/ranking. If GNINA is not installed, the pipeline still runs with a lighter heuristic (no external binary). We recommend using physics-based rescoring when available.
- **PoseBusters:** set `postprocessing.bust_config` to `"redock"` or `"redock-fast"` to run checks, or `null` to skip. If a row used a **reference SDF** for the pocket (cross-docking), it is evaluated with the `**dock`** preset instead of `redock`, even when `bust_config` is `redock`. Notebooks often rank by PoseBusters only for simplicity.

### Installing GNINA (Vina / Vinardo scoring)

To use `postprocessing.scoring: "vina"` or `"vinardo"`, the GNINA binary must be on your `PATH`. Two options:

1. **Automated env (SLURM):** The script `slurm/env_setup.sh` creates a conda environment and optionally installs the GNINA binary (`INSTALL_GNINA=true`). Use it as a reference for a repeatable GNINA install (e.g. on a cluster).

2. **Manual install:** download from [GNINA releases](https://github.com/gnina/gnina/releases) (e.g. [v1.3.2](https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2)), rename to `gnina`, make it executable (`chmod +x gnina`), and add to `PATH` (you may need matching CUDA/cuDNN in the same environment).

### Custom ranking and top-k-from-N

GNINA/Vinardo and PoseBusters are **defaults**, not a requirement for **how** you rank poses or define **top-k-from-N**. With predictions (and optional per-seed scores) under `results/.../seed_*/`, you may use **any** scoring function. `sigmadock.chem.statistics` provides helpers to aggregate across seeds and sort (`collect_posebusters`, `collect_scores`, `sort_statistics_for_top_k`, `compute_top_k_statistics`, `compute_heuristic`); you can also rank directly from the saved `.pt` files.

---

## SLURM

Example scripts live in `slurm/`. See `slurm/README.md` for usage.

**Quick start:**

```bash
# 1. Create env (once)
bash slurm/env_setup.sh

# 2. Training
export DATA_DIR=/path/to/data
sbatch slurm/train.sh

# 3. Sampling (optional: EXPERIMENT=posebusters or astex; omit for CSV / explicit inference.* paths)
export CKPT_DIR=/path/to/checkpoint.ckpt
export DATA_DIR=/path/to/data
export EXPERIMENT=posebusters
sbatch slurm/sample.sh
```

Edit `#SBATCH` directives in each script for your cluster (partition, output paths, etc.).

---

## Notebooks

Notebooks in the `notebooks/` directory give a short, reproducible path from data to metrics. See `notebooks/README.md` for the full list (01–05, extensions).


| Notebook                         | Description                                                                                                          |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Visualize Data & Pocket**      | Load a complex, show protein/ligand and pocket definition (distance cutoff, CoM).                                    |
| **Load model, Sample & Metrics** | Load a checkpoint, run sampling on a small set, visualise trajectories and compute metrics (e.g. RMSD, PoseBusters). |


Run from the repo root so paths and imports match. Optional env vars:

- `SIGMADOCK_DATA_DIR` — data root (default e.g. `./data`).
- `SIGMADOCK_CKPT_DIR` — directory or path to a checkpoint for the sampling notebook.

If GNINA is installed, ranking can use Vina/Vinardo scores via config; otherwise the pipeline uses heuristic physicochemical metrics. The sampling notebook can show ranking by PoseBusters only.

---

## Quick reference


| Task      | Command / location                                                      |
| --------- | ----------------------------------------------------------------------- |
| Install   | `pip install -e ".[train,dev,test]"`                                    |
| Train     | `python scripts/train.py --data_dir <root> --train_exps ...`            |
| Sample    | `python scripts/sample.py ckpt=... data_dir=... experiment=posebusters` |
| Configs   | `conf/experiments/*.yaml`, `conf/sampling/base.yaml`                    |
| Notebooks | `notebooks/` (visualise data/pocket; load model, sample, metrics)       |


---

## Citation

If you use this work in your research, please cite us:

```bibtex
@inproceedings{pratSigmadock2026,
  title     = {SigmaDock: Untwisting Molecular Docking with Fragment-Based SE(3) Diffusion},
  author    = {Prat, Alvaro and Zhang, Leo and Deane, Charlotte and Teh, Yee Whye and Morris, Garrett},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

