# Dataset schema

This repo expects every dataset (DRISHTI-GS, RIM-ONE v3, or your in-house data) to land in this layout after preprocessing:

```
data/<dataset>/
├── images/<stem>.png            # CLAHE'd, 800x800 RGB, OD-centered
├── masks/
│   ├── od/<stem>.png            # 0/255 binary, 800x800
│   └── oc/<stem>.png            # 0/255 binary, 800x800
└── splits/official.json         # {"train": [stems...], "val": [...], "test": [...]}
```

## Raw → preprocessed

`scripts/preprocess.py` reads from `data/<dataset>/raw/` and produces the layout above. The script is intentionally minimal about what it expects in `raw/` — it looks for triples of:

```
<stem>.png
<stem>_od.png
<stem>_oc.png
```

If your source distribution doesn't already use that convention (most don't), add a small adapter step. The DRISHTI-GS provided masks are PNGs in subfolders; the RIM-ONE v3 distribution ships separate `Disc/` and `Cup/` directories. Two adapter snippets you can drop into `scripts/preprocess.py`:

### DRISHTI-GS adapter

```python
# DRISHTI-GS provides per-image folders with `*_OD.png` and `*_Cup.png`
# Walk those, normalize names to <stem>{,_od,_oc}.png, then call process_one().
```

### RIM-ONE v3 adapter

```python
# RIM-ONE v3 ships images in `Stereo Images/` and masks in `Expert*/Disc/` and `Expert*/Cup/`.
# Use Expert 1 by default (paper convention).
```

Both adapters are intentionally not pre-implemented because the public distributions vary by mirror; check what your downloaded zip actually contains and adjust.

## In-house / custom data

Use `configs/data/custom.yaml` and follow the schema above. A minimum viable directory is:

```
data/my_dataset/
├── images/case_001.png
├── masks/od/case_001.png
├── masks/oc/case_001.png
├── ...
└── splits/official.json
```

`splits/official.json` is a single JSON object with three keys (`train`, `val`, `test`) and lists of stems. If you don't have a canonical split, run the deterministic splitter from `scripts/preprocess.py` (`split_by_hash`) and commit the resulting JSON.

## Validation rules

`FundusODOCDataset.__getitem__` will raise `ValueError` if either OD or OC mask for a given stem is empty. Drop such stems from your splits before training.

## Manual download fallback

If `download_data.sh` fails (URLs change), you can place the originals manually:

| Dataset | Where to look |
|---|---|
| DRISHTI-GS | <https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php> |
| RIM-ONE v3 | search "RIM-ONE DL release" on the MIAG-ULL GitHub org; mirror locations rotate |
| In-house | not redistributed |

Then `unzip` into `data/<dataset>/raw/` and run `scripts/preprocess.py`.
