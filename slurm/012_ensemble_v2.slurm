#!/bin/bash
#SBATCH --job-name=rdcnn-v2-ensemble
#SBATCH --partition=gpu-a100-short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
#
# 3-seed ensemble inference on both datasets. No training, just inference
# averaging across the 3 RETFound checkpoints per dataset.
#

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

echo "=== DRISHTI-GS 3-seed ensemble ==="
DRISHTI_CKPTS=$(ls -d outputs/v2_drishti_seed4*_20260513*/best.ckpt 2>/dev/null | sort)
echo "checkpoints:"
echo "$DRISHTI_CKPTS"
srun $PY scripts/ensemble_v2.py \
    --checkpoints $DRISHTI_CKPTS \
    --data-root data/drishti_gs \
    --split test \
    --output-json outputs/v2_drishti_ensemble.json \
    --use-tta

echo ""
echo "=== RIM-ONE v3 3-seed ensemble ==="
RIMONE_CKPTS=$(ls -d outputs/v2_rimone_seed4*_20260513*/best.ckpt 2>/dev/null | sort)
echo "checkpoints:"
echo "$RIMONE_CKPTS"
srun $PY scripts/ensemble_v2.py \
    --checkpoints $RIMONE_CKPTS \
    --data-root data/rim_one_v3 \
    --split test \
    --output-json outputs/v2_rimone_ensemble.json \
    --use-tta

echo ""
echo "=== summary ==="
for f in outputs/v2_drishti_ensemble.json outputs/v2_rimone_ensemble.json; do
  echo "--- $f ---"
  $PY -c "
import json
d = json.load(open('$f'))
best = d['best_recipe']
m = d['results'][best]
print(f'  best recipe: {best}')
print(f'  OD={m[\"od_dice\"]:.4f}  OC={m[\"oc_dice\"]:.4f}  AUC={m[\"glaucoma_auc\"]:.4f}  CDR_MAE={m[\"cdr_mae\"]:.4f}')
"
done
