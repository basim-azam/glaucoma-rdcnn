"""Download all CD-Former checkpoints listed in docs/manifest.json from HuggingFace Hub.

Verifies MD5 sums after download. Requires HF_TOKEN in the environment while
the repo is private; for public repos the token is optional.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
    python scripts/download_weights.py
    # or, to fetch one file only:
    python scripts/download_weights.py --only cd_former_drishti_seed43
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("!! huggingface_hub not installed. Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/manifest.json", type=Path)
    ap.add_argument("--dest", default="weights", type=Path,
                    help="Local directory to download into")
    ap.add_argument("--only", default=None,
                    help="Download only one checkpoint by name (e.g. cd_former_drishti_seed43)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    repo_id = manifest["huggingface_repo"]
    if "basim-azam" in repo_id:
        print(f"!! manifest.json still has placeholder repo_id: {repo_id}", file=sys.stderr)
        print("   Update docs/manifest.json with the actual HF repo before downloading.", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not set. Required for private repos.", file=sys.stderr)

    args.dest.mkdir(parents=True, exist_ok=True)

    weights = manifest["weights"]
    if args.only:
        weights = [w for w in weights if w["name"] == args.only]
        if not weights:
            print(f"!! no checkpoint named {args.only}", file=sys.stderr)
            return 3

    n_ok = 0
    for w in weights:
        dest_path = args.dest / w["filename"]
        if dest_path.exists():
            existing_md5 = _md5_of(dest_path)
            if existing_md5 == w["md5"]:
                print(f"[cached] {w['filename']}")
                n_ok += 1
                continue
        print(f"[downloading] {w['filename']} from {repo_id}")
        cached = hf_hub_download(repo_id=repo_id, filename=w["filename"], token=token)
        # Copy from HF cache to our local dest
        import shutil
        shutil.copy2(cached, dest_path)
        # Verify
        md5 = _md5_of(dest_path)
        if md5 != w["md5"]:
            print(f"!! MD5 mismatch for {w['filename']}: got {md5}, expected {w['md5']}", file=sys.stderr)
            return 4
        print(f"           ✓ MD5 verified")
        n_ok += 1

    print(f"\nDone. {n_ok} / {len(weights)} checkpoints in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
