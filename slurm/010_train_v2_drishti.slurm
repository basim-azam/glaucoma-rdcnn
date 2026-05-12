#!/bin/bash
#SBATCH --job-name=rdcnn-v2-drishti
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
# Usage:
#   sbatch slurm/010_train_v2_drishti.slurm                  # default: 40 ep, seed 42
#   sbatch slurm/010_train_v2_drishti.slurm 43               # custom seed
#   sbatch slurm/010_train_v2_drishti.slurm 42 60            # seed + epochs
#

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

SEED="${1:-42}"
EPOCHS="${2:-40}"
TS=$(date +%Y%m%d-%H%M%S)
RUN_NAME="v2_drishti_seed${SEED}_${TS}"

echo "=== v2 train on DRISHTI-GS ==="
echo "seed=$SEED  epochs=$EPOCHS  run=$RUN_NAME"

# Install v2 deps if needed (idempotent)
$PY -m pip install --quiet --no-warn-script-location \
    'timm>=1.0' 'huggingface_hub>=0.20' 'scipy>=1.11' 'scikit-learn>=1.3' \
    || echo "(some deps already present)"

srun $PY scripts/train_v2.py \
    --data-root data/drishti_gs \
    --output-base outputs \
    --run-name "$RUN_NAME" \
    --epochs "$EPOCHS" \
    --batch-size 16 \
    --freeze-epochs 10 \
    --base-lr 5e-4 \
    --seed "$SEED" \
    --num-workers 8 \

echo "=== done ==="
ls -la "outputs/$RUN_NAME/"
