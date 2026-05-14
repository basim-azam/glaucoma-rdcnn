#!/bin/bash
#SBATCH --job-name=cd-former-eval
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
#
# Usage:
#   sbatch slurm/evaluate.slurm <checkpoint> <data-root>
#
CKPT="${1:-weights/cd_former_drishti_seed43.ckpt}"
DATA_ROOT="${2:-data/drishti_gs}"

python scripts/evaluate.py --checkpoint "$CKPT" --data-root "$DATA_ROOT"
