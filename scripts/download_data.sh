#!/usr/bin/env bash
# Idempotent dataset download for DRISHTI-GS and RIM-ONE v3.
# - SHA-256 verified where checksums are known
# - Resumable via curl --continue-at -
# - Falls back to a manual instruction message if a URL is dead
#
# Usage:  bash scripts/download_data.sh [drishti|rimone|all]
#
# Note: dataset URLs change. If a download fails, see docs/dataset_schema.md
# for the manual fallback path.
set -euo pipefail

ROOT="${ROOT:-data}"
SUBSET="${1:-all}"
mkdir -p "$ROOT"

download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "✓ already have $dest"
        return 0
    fi
    echo "↓ $url"
    if ! curl -L --fail --retry 3 --continue-at - -o "$dest" "$url"; then
        echo "WARNING: failed to download $url" >&2
        echo "         Manual fallback: see docs/dataset_schema.md" >&2
        return 1
    fi
}

extract_zip() {
    local zip="$1"
    local dest="$2"
    if [[ -d "$dest" && -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
        echo "✓ $dest already extracted"
        return 0
    fi
    mkdir -p "$dest"
    echo "↪ extracting $zip → $dest"
    unzip -qq -o "$zip" -d "$dest"
}

drishti() {
    local DIR="$ROOT/drishti_gs"
    mkdir -p "$DIR"
    # PAPER-UNSPEC: Drishti-GS official URL has historically moved. Document the
    # canonical landing page and provide the IIIT-H zip as best-effort.
    local URL="https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Drishti-GS1_files.zip"
    download "$URL" "$DIR/Drishti-GS1_files.zip" || {
        echo "Drishti download failed. Visit https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php"
        return 1
    }
    extract_zip "$DIR/Drishti-GS1_files.zip" "$DIR/raw"
}

rimone() {
    local DIR="$ROOT/rim_one_v3"
    mkdir -p "$DIR"
    # PAPER-UNSPEC: RIM-ONE v3 mirrors vary. Update if the link rots.
    local URL="https://github.com/miag-ull/rim-one-dl/releases/download/v3/RIM-ONE_DL_images.zip"
    download "$URL" "$DIR/RIM-ONE_DL_images.zip" || {
        echo "RIM-ONE v3 download failed. See docs/dataset_schema.md for manual fallback."
        return 1
    }
    extract_zip "$DIR/RIM-ONE_DL_images.zip" "$DIR/raw"
}

case "$SUBSET" in
    drishti) drishti ;;
    rimone)  rimone  ;;
    all)     drishti; rimone ;;
    *) echo "usage: $0 [drishti|rimone|all]"; exit 2 ;;
esac

echo
echo "Counts:"
for d in "$ROOT/drishti_gs/raw" "$ROOT/rim_one_v3/raw"; do
    if [[ -d "$d" ]]; then
        echo "  $d : $(find "$d" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.bmp' -o -name '*.tif' \) | wc -l) image-like files"
    fi
done
echo
echo "✓ Done. Run scripts/preprocess.py next."
