#!/bin/bash
#SBATCH --job-name=rdcnn-eval
#SBATCH --partition=gpu-a100-short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

CKPT="${1:-outputs/best.ckpt}"

srun $PY scripts/evaluate.py \
    --config-name=experiment/repro_drishti \
    +checkpoint="$CKPT"
