"""One-shot upload of all 6 CD-Former checkpoints to HuggingFace Hub.

Run from anywhere you have the .ckpt files and an HF write-token in the
environment. Creates the model repo if it doesn't exist, then uploads each
checkpoint plus a generated README.md model card.

Usage:
    export HF_TOKEN=hf_yyyyyyyyyyyyyyyyyyyyy   # write-scope token
    python scripts/upload_to_huggingface.py \
        --weights-dir /data/.../release_weights \
        --repo-id basim-azam/cd-former-weights \
        --private
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_file
except ImportError:
    print("!! Run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)


_MODEL_CARD = """---
license: cc-by-nc-4.0
library_name: pytorch
tags:
  - medical-imaging
  - fundus
  - glaucoma
  - segmentation
  - retinal-imaging
---

# CD-Former — Joint Optic Disc and Cup Segmentation

CD-Former is a unified four-component framework for joint optic-disc and optic-cup segmentation
from colour fundus photographs. It pairs a fundus-pretrained Masked Autoencoder encoder with a
query-based dense-mask decoder and a geometry-consistent vCDR refinement stage at inference.

This repository hosts the pretrained checkpoints. The model code is at
`<GIT_URL>` (currently private during submission).

## Loading a checkpoint

```python
import torch
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="basim-azam/cd-former-weights",
    filename="cd_former_drishti_seed43.ckpt",
)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(ckpt.keys())  # backbone, head, cfg, ...
```

## Available checkpoints

| Filename | Dataset | OD Dice | OC Dice | Glaucoma AUC | vCDR MAE |
|---|---|--:|--:|--:|--:|
| cd_former_drishti_seed42.ckpt | DRISHTI-GS | 96.41 | 90.16 | 0.990 | 0.052 |
| cd_former_drishti_seed43.ckpt | DRISHTI-GS | 96.75 | 90.57 | 0.990 | 0.062 |
| cd_former_drishti_seed44.ckpt | DRISHTI-GS | 96.23 | 88.14 | 0.990 | 0.063 |
| cd_former_rimone_seed42.ckpt  | RIM-ONE v3 | 95.19 | 75.10 | 0.947 | 0.076 |
| cd_former_rimone_seed43.ckpt  | RIM-ONE v3 | 94.72 | 70.91 | 0.947 | 0.093 |
| cd_former_rimone_seed44.ckpt  | RIM-ONE v3 | 94.82 | 71.59 | 0.961 | 0.087 |

## Citation

To be added at public release.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True, type=Path,
                    help="Directory containing the 6 cd_former_*.ckpt files")
    ap.add_argument("--repo-id", required=True,
                    help="HuggingFace repo id, e.g. basim-azam/cd-former-weights")
    ap.add_argument("--private", action="store_true", help="Create as private repo")
    ap.add_argument("--manifest", default="docs/manifest.json", type=Path)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("!! HF_TOKEN (write scope) required.", file=sys.stderr)
        return 2

    print(f"Target repo: {args.repo_id} ({'private' if args.private else 'public'})")

    # 1. Create the repo (idempotent)
    print("Creating/confirming repo...")
    create_repo(repo_id=args.repo_id, token=token, private=args.private,
                exist_ok=True, repo_type="model")

    # 2. Upload README model card
    readme = _MODEL_CARD.replace("basim-azam", args.repo_id.split("/")[0])
    readme_path = args.weights_dir / "README.md"
    readme_path.write_text(readme)
    upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md",
                repo_id=args.repo_id, token=token, repo_type="model")
    print("  uploaded README.md")

    # 3. Upload each checkpoint
    with open(args.manifest) as f:
        manifest = json.load(f)
    for w in manifest["weights"]:
        ckpt_path = args.weights_dir / w["filename"]
        if not ckpt_path.exists():
            print(f"  !! missing: {ckpt_path}", file=sys.stderr)
            continue
        print(f"  uploading {w['filename']} ({ckpt_path.stat().st_size / 1e9:.2f} GB)...")
        upload_file(path_or_fileobj=str(ckpt_path), path_in_repo=w["filename"],
                    repo_id=args.repo_id, token=token, repo_type="model")
        print(f"    ✓ uploaded")

    # 4. Optionally upload manifest.json so downstream users can introspect
    upload_file(path_or_fileobj=str(args.manifest), path_in_repo="manifest.json",
                repo_id=args.repo_id, token=token, repo_type="model")
    print("  uploaded manifest.json")

    print(f"\nDone. View at: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
