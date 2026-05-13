#!/bin/bash
#SBATCH --job-name=rdcnn-v2-chromatic
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=basimazam0@gmail.com
set -euo pipefail
#
# Chromatic preprocessing variant of CD-Former (R=OD, G=OC, L=fallback).
# Inspired by Dual Opti-TransNet (ESWA 2026) channel-extraction insight.
#
# Usage:
#   sbatch slurm/014_train_v2_chromatic.slurm drishti 42 40
#   sbatch slurm/014_train_v2_chromatic.slurm rimone  42 60
#

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

DATASET="${1:-drishti}"
SEED="${2:-42}"
EPOCHS="${3:-40}"
MODE="${4:-rg_lum}"
TS=$(date +%Y%m%d-%H%M%S)
RUN_NAME="v2_chromatic_${MODE}_${DATASET}_seed${SEED}_${TS}"

if [[ "$DATASET" == "drishti" ]]; then
    DATA_ROOT="data/drishti_gs"
else
    DATA_ROOT="data/rim_one_v3"
fi

echo "=== CD-Former with Chromatic Preprocessing ==="
echo "dataset=$DATASET  seed=$SEED  epochs=$EPOCHS  mode=$MODE  run=$RUN_NAME"

srun $PY scripts/train_v2.py \
    --data-root "$DATA_ROOT" \
    --output-base outputs \
    --run-name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch-size 16 \
    --freeze-epochs 10 \
    --base-lr 5e-4 \
    --seed "$SEED" \
    --num-workers 8 \
    --chromatic-mode "$MODE"

echo "=== done ==="
ls -la "outputs/$RUN_NAME/" 2>/dev/null | head
