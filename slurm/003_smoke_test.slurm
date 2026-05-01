#!/bin/bash
#SBATCH --job-name=rdcnn-smoke
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

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
pytest tests/ -q
# fast_dev_run does one batch through the full pipeline
python scripts/train.py --config-name=experiment/repro_drishti trainer.fast_dev_run=true || echo "(fast_dev_run failed — likely missing preprocessed data; pytest above is the canonical smoke check)"
