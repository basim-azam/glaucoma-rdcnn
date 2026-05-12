#!/bin/bash
#SBATCH --job-name=rdcnn-smoke-v2
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

echo "=== env check ==="
$PY -c "import torch; print(f'torch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name() if torch.cuda.is_available() else \"\"}')"
echo ""
echo "=== installing v2 deps (if needed) ==="
$PY -m pip install --quiet --no-warn-script-location \
    'timm>=1.0' 'huggingface_hub>=0.20' 'scipy>=1.11' \
    || echo "(some deps already present)"
echo ""
echo "=== running v2 smoke test 5 ==="
srun $PY -m pytest tests_v2/test_05_e2e_smoke.py -v -s --tb=short
