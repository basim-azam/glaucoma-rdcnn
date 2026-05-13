#!/bin/bash
#SBATCH --job-name=rdcnn-v2-multirater-real
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
#
# Retrain multirater AFTER per-rater extraction (slurm/015).
# Verifies per-rater masks exist before training, falls back with warning otherwise.

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

SEED="${1:-42}"
EPOCHS="${2:-60}"

echo "=== check per-rater masks ==="
for r in 1 2 3 4; do
    n=$(find data/drishti_gs/masks/od_rater${r} -name '*.png' 2>/dev/null | wc -l)
    echo "  od_rater${r}: $n"
    if [ "$n" -lt 30 ]; then
        echo "!! od_rater${r} has only $n masks — extraction may be incomplete"
    fi
done

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
