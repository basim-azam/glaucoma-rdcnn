#!/bin/bash
#SBATCH --job-name=rdcnn-data
#SBATCH --partition=sapphire
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Three modes (1st arg = dataset name, 2nd arg = optional source path):
#
# 1. AUTO-DOWNLOAD (URLs often dead, falls through to mode 2):
#       sbatch slurm/002_download_data.slurm drishti_gs
#       sbatch slurm/002_download_data.slurm rim_one_v3
#
# 2. MANUAL DIR (you've rsynced an extracted dataset dir):
#       sbatch slurm/002_download_data.slurm drishti_gs /path/to/Drishti-GS1_files
#       sbatch slurm/002_download_data.slurm rim_one_v3 /path/to/RIM-ONE_r3
#
# 3. MANUAL TARBALL (you've rsynced a .tar.gz):
#       sbatch slurm/002_download_data.slurm drishti_gs /path/to/drishti_gs_raw.tar.gz
#       sbatch slurm/002_download_data.slurm rim_one_v3 /path/to/rim_one_v3_raw.tar.gz
#
set -euo pipefail

source /data/gpfs/projects/punim2920/glaucoma-rdcnn/slurm/env_spartan.sh
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn

DATASET="${1:-drishti_gs}"
INPUT="${2:-}"

case "$DATASET" in
    drishti_gs)
        ADAPTER="scripts/prepare_drishti_gs.py"
        ;;
    rim_one_v3)
        ADAPTER="scripts/prepare_rim_one.py"
        ;;
    *)
        echo "!! Unknown dataset: $DATASET" >&2
        echo "   Supported: drishti_gs, rim_one_v3" >&2
        exit 2
        ;;
esac

# --- Resolve the source directory ---------------------------------------
SRC=""
if [[ -z "$INPUT" ]]; then
    bash scripts/download_data.sh "$DATASET" || {
        echo "Auto-download failed. Re-run as:"
        echo "  sbatch $0 $DATASET /path/to/extracted_dir"
        echo "  sbatch $0 $DATASET /path/to/${DATASET}_raw.tar.gz"
        exit 1
    }
    SRC="data/$DATASET/raw"
elif [[ -f "$INPUT" && "$INPUT" =~ \.(tar\.gz|tgz)$ ]]; then
    EXTRACT_TO="data/$DATASET/_extracted"
    mkdir -p "$EXTRACT_TO"
    echo "extracting $INPUT -> $EXTRACT_TO"
    tar -xzf "$INPUT" -C "$EXTRACT_TO"
    SRC="$EXTRACT_TO"
elif [[ -d "$INPUT" ]]; then
    SRC="$INPUT"
else
    echo "!! Don't know how to handle: $INPUT" >&2
    exit 2
fi

# --- Adapt to repo schema -----------------------------------------------
$PY "$ADAPTER" --source-dir "$SRC" --out "data/$DATASET/raw"

# --- CLAHE + 800x800 ROI crop -------------------------------------------
$PY scripts/preprocess.py --config-name=data/$DATASET

echo "OK: $DATASET data ready"
ls -la "data/$DATASET/"
echo "Counts:"
find "data/$DATASET/images" -name '*.png' | wc -l | xargs echo "  images:"
find "data/$DATASET/masks"  -name '*.png' | wc -l | xargs echo "  masks: "
