#!/bin/bash
#SBATCH --job-name=rdcnn-eval
#SBATCH --partition=gpu-a100-short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
#
# Usage:
#   sbatch slurm/008_evaluate.slurm                                  # auto: latest repro_drishti
#   sbatch slurm/008_evaluate.slurm auto repro_rimone                # auto: latest repro_rimone
#   sbatch slurm/008_evaluate.slurm path/to/best.ckpt                # explicit ckpt, drishti
#   sbatch slurm/008_evaluate.slurm path/to/best.ckpt repro_rimone   # explicit ckpt + experiment
#

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

CKPT_IN="${1:-auto}"
EXPERIMENT="${2:-repro_drishti}"

# Auto-discover the newest matching best.ckpt if requested
if [[ "$CKPT_IN" == "auto" ]]; then
    CKPT=$(ls -td outputs/${EXPERIMENT}_*/best.ckpt 2>/dev/null | head -1 || true)
    if [[ -z "$CKPT" ]]; then
        echo "!! No outputs/${EXPERIMENT}_*/best.ckpt found. Pass an explicit path." >&2
        exit 2
    fi
    echo "auto-selected checkpoint: $CKPT"
else
    CKPT="$CKPT_IN"
fi

if [[ ! -f "$CKPT" ]]; then
    echo "!! checkpoint not found: $CKPT" >&2
    exit 3
fi

srun $PY scripts/evaluate.py \
    +experiment="$EXPERIMENT" \
    +checkpoint="$CKPT"
