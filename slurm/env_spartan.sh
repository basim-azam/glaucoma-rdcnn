#!/bin/bash
# Spartan environment for glaucoma-rdcnn (project: punim2920)
# Source this from every interactive session and every sbatch script.
# Pattern adapted from punim2198/ba_data/depth_estimation (proven working).
set -euo pipefail

PROJECT=punim2920
PROJECT_ROOT=/data/gpfs/projects/${PROJECT}/glaucoma-rdcnn
CONDA_ENV=${PROJECT_ROOT}/conda_envs/glaucoma

# --- Modules ---
# CUDA: Spartan ships CUDA 12.2.0 / 12.4.1 / 12.5.1 — no 12.1.x. CUDA 12.2 is
# forward-compatible with the cu121 PyTorch wheels (the wheels bundle their
# own libcudart anyway). Verified via `module spider CUDA` on 2026-05-02.
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.2.0
module load cuDNN/8.9.7.29-CUDA-12.2.0

# --- Conda activate (full path so it works from any cwd) ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- BULLETPROOF python resolution ---
# `conda activate` should put $CONDA_ENV/bin on PATH, but in some sbatch contexts
# the prepend doesn't take. We force it explicitly and clear bash's command-lookup
# cache. After this point, bare `python` is guaranteed to be the env's python.
export PATH="${CONDA_ENV}/bin:${PATH}"
hash -r 2>/dev/null || true
PY="${CONDA_ENV}/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "FATAL: $PY not found — conda env not built? Re-run slurm/001_setup_env.slurm" >&2
    exit 1
fi
export PY

# --- Caches in project dir (home dir quota is tight) ---
export TORCH_HOME=${PROJECT_ROOT}/.cache/torch
export HF_HOME=${PROJECT_ROOT}/.cache/huggingface
export PIP_CACHE_DIR=${PROJECT_ROOT}/.cache/pip
export CONDA_PKGS_DIRS=${PROJECT_ROOT}/.cache/conda_pkgs
mkdir -p "$TORCH_HOME" "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

# --- Block pip --user fallback (causes home-dir quota errors on Spartan) ---
export PIP_USER=no
export PYTHONUSERBASE=${PROJECT_ROOT}/.cache/pyuser
mkdir -p "$PYTHONUSERBASE"

# --- HF token: pull just the export from ~/.bashrc (sbatch doesn't source it). ---
# We grep specifically for the HF_TOKEN line to avoid pulling in anything else
# from ~/.bashrc (e.g. a stale ~/.local PATH addition).
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ -f "$HOME/.bashrc" ]] && grep -q "^export HF_TOKEN=" "$HOME/.bashrc"; then
        eval "$(grep "^export HF_TOKEN=" "$HOME/.bashrc")"
    fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "WARNING: HF_TOKEN not set. Some pretrained downloads may fail." >&2
fi

# --- W&B offline by default ---
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${PROJECT_ROOT}/outputs/wandb

# --- src layout safety net ---
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo "✓ env ready | $($PY --version) | torch $($PY -c 'import torch;print(torch.__version__)') | CUDA: $($PY -c 'import torch;print(torch.cuda.is_available())')"
