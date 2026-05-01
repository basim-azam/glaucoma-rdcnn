#!/bin/bash
#SBATCH --job-name=rdcnn-train-h100
#SBATCH --partition=gpu-h100
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=900G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=basimazam0@gmail.com
# NOTE: gpu-h100 queue often runs 6+ days. Only submit here if you're not in a hurry.
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/train.py \
    --config-name=experiment/repro_drishti \
    training=spartan \
    trainer.devices=4
