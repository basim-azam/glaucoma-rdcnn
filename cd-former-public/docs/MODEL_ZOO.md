# CD-Former Model Zoo

Six RETFound-encoded CD-Former checkpoints, three independent training seeds per dataset. The release is hosted on **HuggingFace Hub** at `basim-azam/cd-former-weights` (private during submission; will be made public alongside the paper).

## Quickstart

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="basim-azam/cd-former-weights",
    filename="cd_former_drishti_seed43.ckpt",
    token="<YOUR_HF_TOKEN>",   # required while the repo is private
)
print(f"Downloaded to: {ckpt_path}")
```

For batch downloading of every checkpoint, use the helper script:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx     # from https://huggingface.co/settings/tokens
bash scripts/download_weights.sh             # or python scripts/download_weights.py
```

## Available checkpoints

| Filename | Dataset | Seed | OD Dice | OC Dice | AUC | vCDR MAE | MD5 |
|---|---|--:|--:|--:|--:|--:|---|
| `cd_former_drishti_seed42.ckpt` | DRISHTI-GS | 42 | 96.41 | 90.16 | 0.990 | 0.052 | `a4fb79ba…f100c1` |
| `cd_former_drishti_seed43.ckpt` | DRISHTI-GS | 43 | 96.75 | 90.57 | 0.990 | 0.062 | `87ac8799…c2d091` |
| `cd_former_drishti_seed44.ckpt` | DRISHTI-GS | 44 | 96.23 | 88.14 | 0.990 | 0.063 | `c792459c…589a7f` |
| `cd_former_rimone_seed42.ckpt`  | RIM-ONE v3 | 42 | 95.19 | 75.10 | 0.947 | 0.076 | `fe522fe7…eeb3e` |
| `cd_former_rimone_seed43.ckpt`  | RIM-ONE v3 | 43 | 94.72 | 70.91 | 0.947 | 0.093 | `1f086afb…0e127` |
| `cd_former_rimone_seed44.ckpt`  | RIM-ONE v3 | 44 | 94.82 | 71.59 | 0.961 | 0.087 | `2648bb94…e44d4` |

Each checkpoint is ~1.3 GB and contains the full state dict (encoder + decoder) plus the training config used to produce it. They are interchangeable in `scripts/evaluate.py` and `scripts/infer.py`.

## Access (currently private)

Weights are shared only with collaborators while the project is under submission. To access:

1. Create a HuggingFace account at https://huggingface.co.
2. Request access from the project authors with your HF username.
3. Once granted, mint a read-token at https://huggingface.co/settings/tokens (scope: "Read").
4. Set `HF_TOKEN=hf_...` in your environment and run the download helper above.

## Public release

License terms for the eventual public release will be set at that time; note that the underlying RETFound encoder is CC BY-NC 4.0 and any future public redistribution must comply with both the CD-Former licence and that of RETFound.
