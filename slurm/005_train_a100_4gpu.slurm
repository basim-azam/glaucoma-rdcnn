#!/bin/bash
#SBATCH --job-name=rdcnn-train-4gpu
#SBATCH --partition=gpu-a100
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=basimazam0@gmail.com
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun ${CONDA_ENV}/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/train.py \
    +experiment=repro_drishti \
    training=spartan \
    trainer.devices=4
