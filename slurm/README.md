# SLURM Scripts

Example SLURM submission scripts for SigmaDock. Customize the `#SBATCH` directives and paths for your cluster.

## Setup

1. **Create conda environment** (run once, from project root):
  ```bash
   bash slurm/env_setup.sh
   # Or submit: sbatch slurm/env_setup.sh
  ```
2. **Edit SBATCH directives** in each script for your cluster (partition, output paths, etc.).

## Training

```bash
export DATA_DIR=/path/to/your/data
sbatch slurm/train.sh
```

Required: `DATA_DIR` — path to the data directory (see main README for data preparation).

## Sampling

```bash
export CKPT_DIR=/path/to/model/checkpoint.ckpt
export DATA_DIR=/path/to/your/data
sbatch slurm/sample.sh
```

Optional: `OUTPUT_DIR` (default: `./sampling_output`).

For array jobs (e.g. multiple seeds):

```bash
sbatch --array=0-39%8 slurm/sample.sh
```

## Environment Variables


| Variable        | Script       | Description                                           |
| --------------- | ------------ | ----------------------------------------------------- |
| `PROJECT_DIR`   | all          | Project root (default: `$SLURM_SUBMIT_DIR` or `.`)    |
| `DATA_DIR`      | train.sh     | Path to data directory (required)                     |
| `DATA_DIR`      | sample.sh    | Path to data directory (default: `$PROJECT_DIR/data`) |
| `CKPT_DIR`      | sample.sh    | Path to model checkpoint (required)                   |
| `OUTPUT_DIR`    | sample.sh    | Sampling output directory                             |
| `CONDA_ENV`     | all          | Conda environment name (default: `sigmadock`)         |
| `EXPERIMENT`    | sample.sh    | Experiment name (default: `posebusters`)               |
| `INSTALL_GNINA` | env_setup.sh | Install gnina for rescoring (default: `true`)         |


