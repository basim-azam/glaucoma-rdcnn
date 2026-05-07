#!/bin/bash
#SBATCH --job-name=rdcnn-rimone-array
#SBATCH --partition=gpu-a100-short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=1-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

CSV=slurm/sweeps/rimone.csv
LINE=$((SLURM_ARRAY_TASK_ID + 1))
ROW=$(sed -n "${LINE}p" "$CSV")
LR=$(echo "$ROW"   | cut -d, -f1)
SEED=$(echo "$ROW" | cut -d, -f2)
BS=$(echo "$ROW"   | cut -d, -f3)

RUN_NAME=sweep_rimone_${SLURM_ARRAY_TASK_ID}_lr${LR}_seed${SEED}_bs${BS}
echo "Sweep row $SLURM_ARRAY_TASK_ID: lr=$LR seed=$SEED bs=$BS  -> $RUN_NAME"

# --- Train ---
srun $PY scripts/train.py \
    +experiment=repro_rimone \
    training=spartan \
    trainer.devices=1 \
    seed=$SEED \
    model.optimizer.lr=$LR \
    data.batch_size=$BS \
    run_name=$RUN_NAME

# --- Eval inline (so each task self-reports its test metrics) ---
echo "=== EVAL: $RUN_NAME ==="
srun $PY scripts/evaluate.py \
    +experiment=repro_rimone \
    +checkpoint="outputs/${RUN_NAME}/best.ckpt"
