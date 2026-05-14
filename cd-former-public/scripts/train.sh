#!/usr/bin/env bash
# Convenience wrapper to train CD-Former on a dataset.
# Usage:
#   ./scripts/train.sh drishti        # 3 seeds, default hyperparams
#   ./scripts/train.sh rimone 42 60   # seed 42, 60 epochs
set -euo pipefail

DATASET="${1:-drishti}"
SEED="${2:-42}"
EPOCHS="${3:-40}"

case "$DATASET" in
    drishti|drishti-gs|drishti_gs) DATA_ROOT="data/drishti_gs"  ;;
    rimone|rim-one|rim_one_v3)     DATA_ROOT="data/rim_one_v3" ;;
    refuge2)                       DATA_ROOT="data/refuge2"    ;;
    smdg19|smdg-19)                DATA_ROOT="data/smdg19"     ;;
    *) echo "Unknown dataset: $DATASET"; exit 2 ;;
esac

python scripts/train.py \
    --data-root "$DATA_ROOT" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size 16
