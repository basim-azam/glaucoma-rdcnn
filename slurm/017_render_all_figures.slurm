#!/bin/bash
#SBATCH --job-name=rdcnn-render-figures
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
# Render every figure referenced in paper/accv/main.tex.
# Outputs land in paper/accv/figures/.
#

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

$PY -m pip install --quiet --no-warn-script-location matplotlib scipy 2>&1 | tail -2 || true

mkdir -p paper/accv/figures

DRISHTI_CKPT=$(ls -d outputs/v2_drishti_seed43_20260513*/best.ckpt | head -1)
RIMONE_CKPT=$(ls -d outputs/v2_rimone_seed42_20260513*/best.ckpt | head -1)

echo "=== fig2 (ablation bar), fig6 (ROC), fig7 (GCvR), fig8 (CDR scatter) ==="
srun $PY scripts/plot_paper_figures.py

echo ""
echo "=== fig4 — qualitative results (DRISHTI top-4) ==="
srun $PY scripts/make_qualitative_figure.py \
    --checkpoint "$DRISHTI_CKPT" \
    --data-root data/drishti_gs \
    --top-k 4 \
    --output-pdf paper/accv/figures/fig4_drishti_qualitative.pdf

echo ""
echo "=== fig4 — qualitative results (RIM-ONE top-3) ==="
srun $PY scripts/make_qualitative_figure.py \
    --checkpoint "$RIMONE_CKPT" \
    --data-root data/rim_one_v3 \
    --top-k 3 \
    --output-pdf paper/accv/figures/fig4_rimone_qualitative.pdf

echo ""
echo "=== fig5 — ablation qualitative (leave-one-out visual) ==="
srun $PY scripts/make_ablation_qualitative.py \
    --checkpoint "$DRISHTI_CKPT" \
    --data-root data/drishti_gs \
    --output-pdf paper/accv/figures/fig5_ablation_qualitative.pdf

echo ""
echo "=== done — all paper figures rendered ==="
ls -la paper/accv/figures/
