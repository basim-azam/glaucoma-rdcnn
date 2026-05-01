#!/bin/bash
#SBATCH --job-name=rdcnn-setup
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

PROJECT_ROOT=/data/gpfs/projects/punim2920/glaucoma-rdcnn
CONDA_ENV=${PROJECT_ROOT}/conda_envs/glaucoma
cd "$PROJECT_ROOT"
mkdir -p logs

# Caches in project dir (NOT $HOME — home is small / file-count quota tight).
export PIP_CACHE_DIR=${PROJECT_ROOT}/.cache/pip
export CONDA_PKGS_DIRS=${PROJECT_ROOT}/.cache/conda_pkgs
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

# CRITICAL: force pip OUT of user-site mode. Spartan's Anaconda3 module + a
# stale ~/.local from prior projects causes pip to silently default to
# --user and dump packages into $HOME (which is at quota). We bypass this by:
#   1. Calling python via the conda env's absolute path (PATH may be shadowed)
#   2. Setting PIP_USER=no explicitly
#   3. Routing PYTHONUSERBASE into the project dir as a defense-in-depth net
export PIP_USER=no
export PYTHONUSERBASE=${PROJECT_ROOT}/.cache/pyuser
mkdir -p "$PYTHONUSERBASE"
unset PYTHONPATH 2>/dev/null || true

module purge
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -d "$CONDA_ENV" ]]; then
    conda create -p "$CONDA_ENV" python=3.10 -y
fi
conda activate "$CONDA_ENV"

# Sanity: confirm we're really using the env's python (not Anaconda3 base or ~/.local)
PY="${CONDA_ENV}/bin/python"
PIP="${PY} -m pip"
echo "=== sanity ==="
echo "python: $PY"
$PY --version
$PY -c "import sys; print('prefix:', sys.prefix); assert sys.prefix.startswith('${CONDA_ENV}'), f'wrong python: {sys.prefix}'"
echo "==============="

$PIP install --upgrade pip wheel
# PyTorch with CUDA 12.1 — proven combo on Spartan (sibling project punim2198).
$PIP install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# Project + dev deps
$PIP install -e ".[dev]"

# Final check
$PY -c "import torch, torchvision; print('torch', torch.__version__, 'cu', torch.version.cuda, 'tv', torchvision.__version__)"
echo "✓ conda env ready at $CONDA_ENV"
