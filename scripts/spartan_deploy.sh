#!/usr/bin/env bash
# Spartan-side deploy. Run this AFTER the GitHub push from bootstrap_repo.sh.
#
#   ssh spartan
#   cd /data/gpfs/projects/punim2920
#   bash <(curl -fsSL https://raw.githubusercontent.com/basim-azam/glaucoma-rdcnn/main/scripts/spartan_deploy.sh)
#
# or, after cloning manually:
#   bash scripts/spartan_deploy.sh
set -euo pipefail

PROJECT=punim2920
PROJECT_ROOT=/data/gpfs/projects/${PROJECT}/glaucoma-rdcnn
GH_USER=${GH_USER:-basim-azam}
REPO=${REPO:-glaucoma-rdcnn}

# --- Clone if not present ---
if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
    cd "/data/gpfs/projects/${PROJECT}"
    git clone "https://github.com/${GH_USER}/${REPO}.git"
fi
cd "$PROJECT_ROOT"
mkdir -p logs

# --- HF token in ~/.bashrc (one-time) ---
# Add to ~/.bashrc manually before running this if you haven't already:
#   echo 'export HF_TOKEN=<your-hf-token>' >> ~/.bashrc
if ! grep -q HF_TOKEN ~/.bashrc 2>/dev/null; then
    echo "WARNING: HF_TOKEN not in ~/.bashrc — pretrained downloads may fail." >&2
    echo "         Add it manually: echo 'export HF_TOKEN=<your-token>' >> ~/.bashrc" >&2
fi

# --- Submit env build ---
SETUP_JOB=$(sbatch --parsable slurm/001_setup_env.slurm)
echo "submitted env setup: $SETUP_JOB"

# --- Submit smoke test depending on env build ---
SMOKE_JOB=$(sbatch --parsable --dependency=afterok:${SETUP_JOB} slurm/003_smoke_test.slurm)
echo "submitted smoke test: $SMOKE_JOB (depends on $SETUP_JOB)"

echo
echo "Track progress with:"
echo "    squeue --me"
echo "    tail -f logs/rdcnn-setup_${SETUP_JOB}.out"
echo "    tail -f logs/rdcnn-smoke_${SMOKE_JOB}.out"
echo
echo "Once smoke passes, kick off real training:"
echo "    sbatch slurm/004_train_a100_1gpu.slurm   # default"
echo "    sbatch slurm/005_train_a100_4gpu.slurm   # DDP, 4 A100s"
