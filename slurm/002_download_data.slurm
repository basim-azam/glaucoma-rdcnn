#!/bin/bash
#SBATCH --job-name=rdcnn-data
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

bash scripts/download_data.sh all
python scripts/preprocess.py --config-name=data/drishti_gs || true
python scripts/preprocess.py --config-name=data/rim_one_v3 || true
echo "✓ data step done"
