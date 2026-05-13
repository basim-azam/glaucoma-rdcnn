#!/bin/bash
#SBATCH --job-name=rdcnn-v2-multirater
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
# Heteroscedastic multi-rater training on DRISHTI-GS.
#
# Usage:
#   sbatch slurm/013_train_v2_multirater.slurm           # default: seed 42, 60 ep
#   sbatch slurm/013_train_v2_multirater.slurm 43        # custom seed
#

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

SEED="${1:-42}"
EPOCHS="${2:-60}"

echo "=== multi-rater training on DRISHTI-GS ==="
echo "seed=$SEED epochs=$EPOCHS"
echo ""
echo "=== checking for per-rater masks ==="
ls data/drishti_gs/masks/ 2>/dev/null || true
if [[ ! -d data/drishti_gs/masks/od_rater1 ]]; then
    echo "!! per-rater masks not yet extracted."
    echo "   Trainer will fall back to merged masks (heteroscedastic head still trains)."
fi

srun $PY scripts/train_v2_multirater.py \
    --data-root data/drishti_gs \
    --epochs "$EPOCHS" \
    --batch-size 16 \
    --freeze-epochs 10 \
    --hetero-warmup-epochs 10 \
    --agreement-weight 0.5 \
    --w-hetero 0.4 \
    --base-lr 5e-4 \
    --seed "$SEED" \
    --num-workers 8
