#!/bin/bash
#SBATCH --job-name=rdcnn-qualitative
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
# Render Fig. 4 of the paper: cherry-picked qualitative results
# for both DRISHTI-GS and RIM-ONE v3, using best RETFound seed.

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

$PY -m pip install --quiet --no-warn-script-location matplotlib 2>&1 | tail -1 || true

mkdir -p paper/accv/figures

echo "=== DRISHTI (seed 43, our strongest single) ==="
DRISHTI_CKPT=$(ls -d outputs/v2_drishti_seed43_20260513*/best.ckpt | head -1)
srun $PY scripts/make_qualitative_figure.py \
    --checkpoint "$DRISHTI_CKPT" \
    --data-root data/drishti_gs \
    --top-k 4 \
    --output-pdf paper/accv/figures/fig4_drishti_qualitative.pdf

echo ""
echo "=== RIM-ONE (seed 42, our strongest single) ==="
RIMONE_CKPT=$(ls -d outputs/v2_rimone_seed42_20260513*/best.ckpt | head -1)
srun $PY scripts/make_qualitative_figure.py \
    --checkpoint "$RIMONE_CKPT" \
    --data-root data/rim_one_v3 \
    --top-k 3 \
    --output-pdf paper/accv/figures/fig4_rimone_qualitative.pdf

echo ""
echo "=== done ==="
ls -la paper/accv/figures/
