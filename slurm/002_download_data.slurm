#!/bin/bash
#SBATCH --job-name=rdcnn-data
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Two modes:
#
# 1. AUTO-DOWNLOAD (the URL is often dead — fall back to mode 2):
#       sbatch slurm/002_download_data.slurm
#
# 2. MANUAL (you've rsynced an extracted/tarred Drishti-GS1_files dir):
#       sbatch slurm/002_download_data.slurm /path/to/Drishti-GS1_files
#       sbatch slurm/002_download_data.slurm /path/to/drishti_gs_raw.tar.gz
#
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

INPUT="${1:-}"

# --- Resolve the source directory ---------------------------------------
SRC=""
if [[ -z "$INPUT" ]]; then
    # Mode 1: try the auto-download (often 404s)
    bash scripts/download_data.sh drishti || {
        echo "Auto-download failed. Re-run as:"
        echo "  sbatch $0 /path/to/Drishti-GS1_files"
        exit 1
    }
    SRC="data/drishti_gs/raw"
elif [[ -f "$INPUT" && "$INPUT" =~ \.(tar\.gz|tgz)$ ]]; then
    # Mode 2a: a tar.gz the user rsynced over
    EXTRACT_TO=data/drishti_gs/_extracted
    mkdir -p "$EXTRACT_TO"
    echo "extracting $INPUT -> $EXTRACT_TO"
    tar -xzf "$INPUT" -C "$EXTRACT_TO"
    SRC="$EXTRACT_TO"
elif [[ -d "$INPUT" ]]; then
    # Mode 2b: a directory the user rsynced over
    SRC="$INPUT"
else
    echo "!! Don't know how to handle: $INPUT" >&2
    echo "   Pass either a directory or a .tar.gz file" >&2
    exit 2
fi

# --- Adapt to repo schema -----------------------------------------------
$PY scripts/prepare_drishti_gs.py --source-dir "$SRC" --out data/drishti_gs/raw

# --- CLAHE + 800x800 ROI crop -------------------------------------------
$PY scripts/preprocess.py --config-name=data/drishti_gs

echo "OK: data ready"
ls -la data/drishti_gs/
echo "Counts:"
find data/drishti_gs/images -name '*.png' | wc -l | xargs echo "  images:"
find data/drishti_gs/masks  -name '*.png' | wc -l | xargs echo "  masks: "
