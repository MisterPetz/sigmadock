#!/bin/bash -l
#
# SigmaDock sampling job. Customize SBATCH directives and variables for your cluster.
#
# Usage: sbatch slurm/sample.sh
# For array jobs: sbatch --array=0-39%8 slurm/sample.sh
#
# Required: Set CKPT_DIR to your model checkpoint path.
# Optional: PROJECT_DIR, DATA_DIR, OUTPUT_DIR, CONDA_ENV, EXPERIMENT (if set, e.g. posebusters or astex; unset = custom inference — set inference.inference_datafront or inference.ligand_sdf+inference.protein_pdb in extra args / yaml)
#
# ------------------------------- SBATCH (customize for your cluster) -------------------------------
#SBATCH --job-name=sigmadock-sampling
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
# For array jobs, uncomment:
# #SBATCH --array=0-39

# ------------------------------- Configuration -------------------------------
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
CKPT_DIR="${CKPT_DIR:?Set CKPT_DIR to your model checkpoint path}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/sampling_output}"
CONDA_ENV="${CONDA_ENV:-sigmadock}"
EXPERIMENT="${EXPERIMENT:-}"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

cd "${PROJECT_DIR}" || exit 1
mkdir -p slurm_logs

# ------------------------------- Conda -------------------------------
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" || { echo "ERROR: failed to activate ${CONDA_ENV}"; exit 1; }
fi

# ------------------------------- Run sampling -------------------------------
# num_seeds=1 for reproducibility. Add experiment=posebusters (or astex, etc.) for a named benchmark.
SAMPLING_ARGS=(
  ckpt="${CKPT_DIR}"
  data_dir="${DATA_DIR}"
  run_tag="sampling"
  graph.sample_conformer=true
  seed="${TASK_ID}"
  output_dir="${OUTPUT_DIR}"
  hardware.devices=auto
  num_seeds=1
  graph.fragmentation_strategy=canonical
  hydra.run.dir="${OUTPUT_DIR}/hydra_out"
)
if [[ -n "${EXPERIMENT}" ]]; then
  SAMPLING_ARGS+=(experiment="${EXPERIMENT}" experiments.sdf_regex=".*ligands.sdf$")
fi

python scripts/sample.py "${SAMPLING_ARGS[@]}"
