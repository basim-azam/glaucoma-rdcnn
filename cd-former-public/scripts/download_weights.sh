#!/usr/bin/env bash
# Download all CD-Former checkpoints from HuggingFace Hub.
# Requires HF_TOKEN in environment while the repo is private.
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Warning: HF_TOKEN not set. If the repo is still private, downloads will fail."
    echo "Mint a token at https://huggingface.co/settings/tokens and:"
    echo "    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
fi

python3 scripts/download_weights.py "$@"
