# SigmaDock

Official implementation of:

> **SigmaDock: Untwisting Molecular Docking with Fragment-Based SE(3) Diffusion**  
> Alvaro Prat, Leo Zhang, Charlotte Dean, Yee Whye Teh, Garrett Morris  
> *International Conference on Learning Representations (ICLR), 2026*

This repository supports **training**, **sampling**, and **evaluation** for the ICLR submission.

**Contact:** [alvaro.prat@stats.ox.ac.uk](mailto:alvaro.prat@stats.ox.ac.uk) · or open an issue. We will reply as soon as possible!

> **Note:** This is a beta release. APIs and behaviour may change in future versions.

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

- **Python** ≥3.9, <3.12  
- **CUDA** (for GPU training/sampling)

### From source (recommended)

```bash
git clone https://github.com/alvaroprat97/sigmadock.git
cd sigmadock

conda create -y -n sigmadock python=3.11
conda activate sigmadock

pip install -e ".[train,dev,test]"
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


| File type            | Required | Description                                                                                                                                                                                                                               |
| -------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Protein**          | Yes      | One `.pdb` file matching the experiment’s `pdb_regex` (e.g. pocket or full structure).                                                                                                                                                    |
| **Ligand(s)**        | Yes      | One or more ligand files (**.sdf**) matching the experiment’s `sdf_regex`. These are the molecules to dock. If an SDF has no valid, sanitized ligands, the loader will try a `**.mol2`** file with the same base name in the same folder. |
| **Reference ligand** | Optional | One **.sdf** matching `ref_sdf_regex` (if set). Used to define the pocket and centre-of-mass when it differs from the ligand being docked (e.g. **cross-docking**). Similar to the **--autobox** option.                                  |


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
- `**ref_sdf_regex`** — *(optional)* pattern for the **reference** ligand SDF used only for pocket definition and CoM. Omit for re-docking (same file as ligand); set for cross-docking so pocket is defined from the native pose.

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
  - Optional whitelist: `<data_root>/posebusters_paper/posebusters_correct_ids.txt` (one PDB/system ID per line). Set `sampling.data.blacklist` to this file to run only on those IDs.

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
python scripts/sample.py \
  sampling.experiments.name=posebusters \
  sampling.data.data_dir=/path/to/data \
  sampling.model.ckpt_dir=/path/to/downloaded_checkpoint.ckpt
```

If you use the SLURM sampling script in `slurm/sample.sh`, set:

```bash
export CKPT_DIR=/path/to/downloaded_checkpoint.ckpt
sbatch slurm/sample.sh
```

---

## Sampling

Sampling uses **Hydra** and the config under `conf/sampling/base.yaml`. Run from the project root so relative paths resolve correctly:

```bash
python scripts/sample.py \
  sampling.experiments.name=posebusters \
  sampling.data.data_dir=/path/to/data \
  sampling.model.ckpt_dir=/path/to/checkpoint.ckpt \
  sampling.data.batch_size=16 # Depending on your GPU-RAM
```

Or with a YAML override:

```bash
python scripts/sample.py --config-name sampling/base --config-path conf/
```

**Post-processing:**

- **Scoring**: In `conf/sampling/base.yaml`, set `postprocessing.scoring` to `"vina"` or `"vinardo"` to use **GNINA** for rescoring/ranking poses. If GNINA is not installed, the pipeline still runs and selects the best pose using heuristic, cheaper physicochemical metrics (no external binary required).
- **PoseBusters**: Set `postprocessing.bust_config` to `"redock"` or `"redock-fast"` to run PoseBusters checks. Notebooks can show ranking by PoseBusters only.

**Installing GNINA (optional, for Vina/Vinardo rescoring)**

To use `postprocessing.scoring: "vina"` or `"vinardo"`, the GNINA binary must be on your `PATH`. Two options:

1. **Manual install**: Download the GNINA binary from [GNINA releases](https://github.com/gnina/gnina/releases) (e.g. [v1.3.2](https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2)), rename it to `gnina`, make it executable (`chmod +x gnina`), and place it in a directory on your `PATH` (e.g. `~/bin` or your conda env’s `bin/`). If you use a conda env, you may need cuDNN/CUDA in that env and `LD_LIBRARY_PATH` set so `gnina` can run; see the script below for a full setup.
2. **Automated env setup (SLURM)**: The script `slurm/env_setup.sh` creates a conda environment and optionally installs the GNINA binary (`INSTALL_GNINA=true`). Use it as a reference for a repeatable GNINA install (e.g. on a cluster).

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

# 3. Sampling
export CKPT_DIR=/path/to/checkpoint.ckpt
export DATA_DIR=/path/to/data
sbatch slurm/sample.sh
```

Edit `#SBATCH` directives in each script for your cluster (partition, output paths, etc.).

---

## Notebooks

Notebooks in `**notebooks/**` give a short, reproducible path from data to metrics.


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


| Task      | Command / location                                                                |
| --------- | --------------------------------------------------------------------------------- |
| Install   | `pip install -e ".[train,dev,test]"`                                              |
| Train     | `python scripts/train.py --data_dir <root> --train_exps ...`                      |
| Sample    | `python scripts/sample.py sampling.data.data_dir=... sampling.model.ckpt_dir=...` |
| Configs   | `conf/experiments/*.yaml`, `conf/sampling/base.yaml`                              |
| Notebooks | `notebooks/` (visualise data/pocket; load model, sample, metrics)                 |


---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{pratSigmadock2026,
  title     = {SigmaDock: Untwisting Molecular Docking with Fragment-Based SE(3) Diffusion},
  author    = {Prat, Alvaro and Zhang, Leo and Deane, Charlotte and Teh, Yee Whye and Morris, Garrett},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

