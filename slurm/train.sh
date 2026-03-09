#!/bin/bash -l
#
# SigmaDock training job. Customize SBATCH directives and variables for your cluster.
#
# Usage: sbatch slurm/train.sh
# Set DATA_DIR and optionally PROJECT_DIR before submitting, or edit below.
#
# ------------------------------- SBATCH (customize for your cluster) -------------------------------
#SBATCH --job-name=sigmadock
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128Gb
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# ------------------------------- Configuration -------------------------------
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to your data directory (e.g. /path/to/sigmadock/data)}"
CONDA_ENV="${CONDA_ENV:-sigmadock}"

# WANDB: set your key in the environment or uncomment and set below
# export WANDB_API_KEY="your-key-here"

cd "${PROJECT_DIR}" || exit 1
mkdir -p slurm_logs

# ------------------------------- Conda -------------------------------
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" || { echo "ERROR: failed to activate ${CONDA_ENV}"; exit 1; }
fi

# ------------------------------- Run training -------------------------------
NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[[ "$NGPUS" -lt 1 ]] && NGPUS=1
NUM_WORKERS=$((${SLURM_CPUS_PER_TASK:-8} - 3))

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NGPUS \
  scripts/train.py \
  --config conf/training/slurm.yaml \
  --data_dir "${DATA_DIR}" \
  --num_workers "${NUM_WORKERS}" \