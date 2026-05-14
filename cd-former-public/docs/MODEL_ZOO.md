# CD-Former Model Zoo

Six RETFound-encoded CD-Former checkpoints, three independent training seeds per dataset. All checkpoints contain the full state dict (backbone + decoder) and the training config; load via `scripts/load_checkpoint.py` or directly with `torch.load()`.

## OneDrive download links

Replace each `<ONEDRIVE_LINK>` below with the direct-download URL once you have uploaded the corresponding file. Use the `bash scripts/download_weights.sh` helper to fetch + verify all six in one shot.

| Checkpoint | URL | MD5 | Size |
|---|---|---|--:|
| `cd_former_drishti_seed42.ckpt` | [`<ONEDRIVE_LINK>`](<ONEDRIVE_LINK>) | `<MD5_DRISHTI_42>` | 1.3 GB |
| `cd_former_drishti_seed43.ckpt` | [`<ONEDRIVE_LINK>`](<ONEDRIVE_LINK>) | `<MD5_DRISHTI_43>` | 1.3 GB |
| `cd_former_drishti_seed44.ckpt` | [`<ONEDRIVE_LINK>`](<ONEDRIVE_LINK>) | `<MD5_DRISHTI_44>` | 1.3 GB |
| `cd_former_rimone_seed42.ckpt`  | [`<ONEDRIVE_LINK>`](<ONEDRIVE_LINK>) | `<MD5_RIMONE_42>`  | 1.3 GB |
| `cd_former_rimone_seed43.ckpt`  | [`<ONEDRIVE_LINK>`](<ONEDRIVE_LINK>) | `<MD5_RIMONE_43>`  | 1.3 GB |
| `cd_former_rimone_seed44.ckpt`  | [`<ONEDRIVE_LINK>`](<ONEDRIVE_LINK>) | `<MD5_RIMONE_44>`  | 1.3 GB |

## Manifest (machine-readable)

[`docs/manifest.json`](manifest.json) lists every artefact with its URL, MD5, and intended use. The download script consumes it directly.

## Access (currently private)

Weights are shared only with collaborators while the project is under submission. Request the OneDrive share from the authors. License terms for the eventual public release will be set at that time; note that the underlying RETFound encoder is CC BY-NC 4.0 and any future public redistribution must comply with both the CD-Former licence and that of RETFound.
