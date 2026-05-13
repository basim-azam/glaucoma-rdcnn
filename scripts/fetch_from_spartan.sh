#!/usr/bin/env bash
# Pull CD-Former results from Spartan to the local repo for figure generation.
# Run from the repo root on a machine with ssh access to spartan.
#
# Usage:
#   ./scripts/fetch_from_spartan.sh                # default: all v2 outputs
#   ./scripts/fetch_from_spartan.sh seed4_20260513 # restrict to today's RETFound runs
#
set -euo pipefail

PATTERN="${1:-}"
SPARTAN_REPO="/data/gpfs/projects/punim2920/glaucoma-rdcnn"
LOCAL_DEST="outputs"

mkdir -p "$LOCAL_DEST/_fetched_from_spartan"

echo "=== fetching summary.json + postproc_test.json for all v2 runs ==="
# Only pull tiny JSONs, not the heavy .ckpt files
rsync -avz --include='*/' \
            --include='summary.json' \
            --include='postproc_test.json' \
            --include='eval_*.json' \
            --exclude='*.ckpt' \
            --exclude='*' \
            spartan:"$SPARTAN_REPO/outputs/v2_${PATTERN}*/" \
            "$LOCAL_DEST/_fetched_from_spartan/" || true

echo ""
echo "=== fetching test images + masks for qualitative figure (DRISHTI + RIM-ONE) ==="
mkdir -p data/drishti_gs/images data/drishti_gs/masks/od data/drishti_gs/masks/oc
mkdir -p data/rim_one_v3/images data/rim_one_v3/masks/od data/rim_one_v3/masks/oc
mkdir -p data/drishti_gs/splits data/rim_one_v3/splits

rsync -avz spartan:"$SPARTAN_REPO/data/drishti_gs/images/" data/drishti_gs/images/
rsync -avz spartan:"$SPARTAN_REPO/data/drishti_gs/masks/" data/drishti_gs/masks/
rsync -avz spartan:"$SPARTAN_REPO/data/drishti_gs/splits/" data/drishti_gs/splits/
rsync -avz spartan:"$SPARTAN_REPO/data/rim_one_v3/images/" data/rim_one_v3/images/
rsync -avz spartan:"$SPARTAN_REPO/data/rim_one_v3/masks/" data/rim_one_v3/masks/
rsync -avz spartan:"$SPARTAN_REPO/data/rim_one_v3/splits/" data/rim_one_v3/splits/

echo ""
echo "=== fetching the BEST RETFound checkpoint per dataset (for qualitative + ROC) ==="
DRISHTI_BEST="v2_drishti_seed43_20260513"
RIMONE_BEST="v2_rimone_seed42_20260513"
mkdir -p outputs
rsync -avz spartan:"$SPARTAN_REPO/outputs/${DRISHTI_BEST}*" outputs/ 2>/dev/null || true
rsync -avz spartan:"$SPARTAN_REPO/outputs/${RIMONE_BEST}*"  outputs/ 2>/dev/null || true

echo ""
echo "=== summary ==="
find "$LOCAL_DEST/_fetched_from_spartan" -name '*.json' | wc -l | xargs echo "  json files:"
find data/drishti_gs/images -name '*.png' | wc -l | xargs echo "  drishti images:"
find data/rim_one_v3/images -name '*.png' | wc -l | xargs echo "  rim-one images:"
find outputs -name 'best.ckpt' | wc -l | xargs echo "  best.ckpt files:"
echo ""
echo "Done. Next: python scripts/plot_paper_figures.py"
