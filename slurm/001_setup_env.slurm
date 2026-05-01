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

module purge
module load Anaconda3/2024.02-1
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the env in the project dir (NOT in ~/anaconda3/envs — home is full).
if [[ ! -d "$CONDA_ENV" ]]; then
    conda create -p "$CONDA_ENV" python=3.10 -y
fi
conda activate "$CONDA_ENV"

pip install --upgrade pip wheel

# PyTorch with CUDA 12.1 — proven combo on Spartan (sibling project punim2198).
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Project + dev deps
pip install -e ".[dev]"

# Sanity check
python -c "import torch, torchvision; print('torch', torch.__version__, 'cu', torch.version.cuda, 'tv', torchvision.__version__)"
echo "✓ conda env ready at $CONDA_ENV"
