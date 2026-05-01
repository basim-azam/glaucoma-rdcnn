#!/bin/bash
#SBATCH --job-name=rdcnn-array
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=1-6           # adjust to match the number of rows below the header in repro.csv
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

CSV=slurm/sweeps/repro.csv
LINE=$((SLURM_ARRAY_TASK_ID + 1))    # +1 to skip header
ROW=$(sed -n "${LINE}p" "$CSV")
LR=$(echo "$ROW" | cut -d, -f1)
SEED=$(echo "$ROW" | cut -d, -f2)
BS=$(echo "$ROW" | cut -d, -f3)

echo "Sweep row $SLURM_ARRAY_TASK_ID: lr=$LR seed=$SEED batch_size=$BS"

srun $PY scripts/train.py \
    +experiment=repro_drishti \
    training=spartan \
    trainer.devices=1 \
    seed=$SEED \
    model.optimizer.lr=$LR \
    data.batch_size=$BS \
    run_name=sweep_${SLURM_ARRAY_TASK_ID}_lr${LR}_seed${SEED}_bs${BS}
