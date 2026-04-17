#!/bin/bash -l
#
# Create conda environment for SigmaDock (training and sampling).
# Run directly: bash slurm/env_setup.sh
# Or submit as job: sbatch slurm/env_setup.sh
#
# Set CONDA_ENV to customize env name (default: sigmadock).
# Set INSTALL_GNINA=false to skip gnina (needed for sampling rescoring).
#
# ------------------------------- SBATCH (optional - for cluster runs) -------------------------------
#SBATCH --job-name=sigmadock-env
#SBATCH --nodes=1
#SBATCH --gpus=0
#SBATCH --mem=8Gb
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}"
CONDA_ENV="${CONDA_ENV:-sigmadock_auto}"
INSTALL_GNINA="${INSTALL_GNINA:-true}"

cd "${PROJECT_DIR}" || exit 1
mkdir -p slurm_logs

# ------------------------------- Conda env -------------------------------
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^${CONDA_ENV} "; then
  echo "Conda env ${CONDA_ENV} already exists. Activating..."
  conda activate "${CONDA_ENV}" || exit 1
else
  echo "Creating conda env '${CONDA_ENV}'..."
  conda create -y -n "${CONDA_ENV}" python=3.11 || { echo "ERROR: Conda create failed"; exit 1; }
  
  echo "Activating '${CONDA_ENV}'..."
  conda activate "${CONDA_ENV}" || exit 1
  
  echo "Running install.sh to install PyTorch and SigmaDock..."
  bash install.sh cu128 || { echo "ERROR: install.sh failed"; exit 1; }
fi


# ------------------------------- Gnina (for sampling rescoring) -------------------------------
if [[ "${INSTALL_GNINA}" == "true" ]]; then
  echo "Installing gnina for sampling rescoring..."
  GNINA_URL="${GNINA_URL:-https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2}"
  wget -q "${GNINA_URL}" -O gnina.download && mv gnina.download gnina && chmod +x gnina
  mkdir -p "${CONDA_PREFIX}/bin"
  cp gnina "${CONDA_PREFIX}/bin/gnina" && rm -f gnina
  echo "Installed gnina to ${CONDA_PREFIX}/bin/"
fi

echo "Done. Activate with: conda activate ${CONDA_ENV}"
