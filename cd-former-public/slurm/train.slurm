#!/bin/bash
#SBATCH --job-name=cd-former-train
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
#
# Usage:
#   sbatch slurm/train.slurm <dataset> <seed> <epochs>
#   sbatch slurm/train.slurm drishti_gs 42 40
#
DATASET="${1:-drishti_gs}"
SEED="${2:-42}"
EPOCHS="${3:-40}"

python scripts/train.py \
    --data-root "data/${DATASET}" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size 16
