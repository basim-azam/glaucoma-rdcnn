#!/bin/bash
#SBATCH --job-name=drishti-raters
#SBATCH --partition=sapphire
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

SRC="${1:-data/drishti_gs/_extracted}"

$PY scripts/extract_drishti_per_rater.py --source-dir "$SRC"
echo ""
echo "=== verify rater dirs ==="
for r in 1 2 3 4; do
    for k in od oc; do
        count=$(find data/drishti_gs/masks/${k}_rater${r} -name '*.png' 2>/dev/null | wc -l)
        echo "  ${k}_rater${r}: $count masks"
    done
done
