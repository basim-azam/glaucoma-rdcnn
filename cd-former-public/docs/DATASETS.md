# Datasets

CD-Former is evaluated on four public datasets. Two options for each:

- **Preprocessed (recommended for retraining):** download our preprocessed copy from the link in `docs/MODEL_ZOO.md`. Already CLAHE-normalised and ROI-cropped, official splits applied.
- **Original (for end users wanting to start from the original archive):** download from the source below and run the adapter + preprocess pipeline.

## DRISHTI-GS

- Source: https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php
- 101 colour fundus images, 4 expert annotations per training image.
- Public release contains averaged (SoftMap) annotations only.
- Adapter: `python scripts/prepare_drishti_gs.py --source-dir /path/to/Drishti-GS1_files`
- Split: official train (50) / test (51).

## RIM-ONE v3

- Source: http://medimrg.webs.ull.es/research/retinal-imaging/rim-one
- 159 stereoscopic fundus images (85 healthy, 74 glaucoma/suspect).
- Public release includes Expert 1 and Expert 2 masks.
- Adapter: `python scripts/prepare_rim_one.py --source-dir /path/to/RIM-ONE_r3`
- Split: held-out 80/20 patient-level (see `splits/official.json` after adapter run).

## REFUGE2

- Source: https://refuge.grand-challenge.org/
- 1,200 fundus images (800 training, 400 validation, 400 test).
- Three-class masks (background / disc / cup) at the original 1634×1634 resolution.
- Adapter: `python scripts/prepare_refuge2.py --source-dir /path/to/REFUGE2`
- Split: official challenge train/test.

## SMDG-19

- Source: https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset
- Aggregate of 19 public fundus datasets; ~12,000 images.
- Adapter: `python scripts/prepare_smdg19.py --source-dir /path/to/SMDG-19 --exclude drishti-gs,rim-one`
- Split: 80/20 random within each source dataset (excluding DRISHTI-GS and RIM-ONE if already evaluated separately).

## Preprocessing

After adapter:

```bash
python scripts/preprocess.py --config-name=data/<dataset>
```

This applies CLAHE on the green channel followed by an 800×800 ROI crop centered on the disc.
