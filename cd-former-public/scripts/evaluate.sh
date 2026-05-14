#!/usr/bin/env bash
# Convenience wrapper: download weights if missing, then evaluate on a dataset.
# Usage:
#   ./scripts/evaluate.sh drishti
#   ./scripts/evaluate.sh rimone
set -euo pipefail

DATASET="${1:-drishti}"

case "$DATASET" in
    drishti|drishti-gs|drishti_gs)
        DATA_ROOT="data/drishti_gs"
        CKPT="weights/cd_former_drishti_seed43.ckpt"
        ;;
    rimone|rim-one|rim_one_v3)
        DATA_ROOT="data/rim_one_v3"
        CKPT="weights/cd_former_rimone_seed42.ckpt"
        ;;
    *) echo "Unknown dataset: $DATASET"; exit 2 ;;
esac

if [[ ! -f "$CKPT" ]]; then
    echo "Checkpoint not found; running download_weights.sh first."
    bash scripts/download_weights.sh
fi

python scripts/evaluate.py --checkpoint "$CKPT" --data-root "$DATA_ROOT"
