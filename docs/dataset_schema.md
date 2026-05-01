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

The IIIT-H download URL for DRISHTI-GS frequently 404s. Use the manual flow:

| Dataset | Where to look |
|---|---|
| DRISHTI-GS | <https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php> (login required) |
| RIM-ONE v3 | search "RIM-ONE DL release" on the MIAG-ULL GitHub org; mirror locations rotate |
| In-house | not redistributed |

### Manual flow for DRISHTI-GS (recommended)

You'll have downloaded `Drishti-GS1_files.rar` (or .zip) locally — license terms typically forbid redistribution, so don't push it to GitHub. Move it to Spartan via rsync/scp instead:

**On Windows** (one-time):

1. Extract the rar with WinRAR (right-click → Extract Here). You should now have a `Drishti-GS1_files/` folder containing `Training/` and `Test/`.
2. Re-pack as tar.gz so Spartan's GNU tar can read it without an unrar module:

   ```powershell
   cd C:\Users\basim\Downloads\glaucoma-rdcnn
   tar -czf drishti_gs_raw.tar.gz Drishti-GS1_files
   ```

   (Windows 10+ ships `tar.exe` natively. If you don't have it, `7z a -ttar - Drishti-GS1_files | 7z a -tgzip -si drishti_gs_raw.tar.gz` works too.)

3. rsync to Spartan:

   ```bash
   rsync -avP drishti_gs_raw.tar.gz spartan:/data/gpfs/projects/punim2920/glaucoma-rdcnn/
   ```

**On Spartan**:

```bash
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn
sbatch slurm/002_download_data.slurm /data/gpfs/projects/punim2920/glaucoma-rdcnn/drishti_gs_raw.tar.gz
```

The job will: extract the tar.gz, run `scripts/prepare_drishti_gs.py` to convert the official Training/Test/SoftMap layout into the repo's `<stem>{,_od,_oc}.png` triples, then run `scripts/preprocess.py` for CLAHE + 800x800 ROI crops.

### Why not put the rar in git or GitHub Releases

- GitHub rejects single files >100MB on plain push. The DRISHTI rar is ~335MB.
- DRISHTI-GS license is "for non-commercial research use" with no explicit redistribution clause; uploading to a public repo or release asset is risky.
- Cloning a 335MB blob from git on every Spartan reset wastes both your time and your home-dir cache.

rsync straight to the project allocation is one command, doesn't pollute git history, and respects the dataset terms.
