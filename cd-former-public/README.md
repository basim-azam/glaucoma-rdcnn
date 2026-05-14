<div align="center">

# CD-Former

### A Unified Framework for Joint Optic Disc and Cup Segmentation with Geometry-Consistent Cup-to-Disc Ratio Estimation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)](https://pytorch.org/)

</div>

---

CD-Former is a single-pass framework for joint optic-disc (OD) and optic-cup (OC) segmentation from colour fundus photographs. It decomposes the task into four cooperating components and produces dense free-form masks together with a clinically meaningful vertical cup-to-disc ratio (vCDR).

```
fundus image  →  DAVE  →  MSFL  →  C2QD  →  GCvR  →  (OD mask, OC mask, vCDR)
```

| Dataset      | OD Dice   | OC Dice   | Glaucoma AUC | vCDR MAE |
|--------------|----------:|----------:|-------------:|---------:|
| DRISHTI-GS   | **96.75** | **90.57** | **0.990**    | **0.052** |
| DRISHTI-GS (3-seed ensemble) | 96.65 | 90.23 | **1.000** | 0.054 |
| RIM-ONE v3   | **95.31** | **75.36** | **0.961**    | **0.076** |

All numbers are reported on the official test splits.

---

## Quickstart

```bash
# Repository access (currently private) — request access from the authors
git clone git@github.com:<USER>/cd-former.git
cd cd-former
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Download pre-trained weights (see MODEL_ZOO.md for direct links)
bash scripts/download_weights.sh

# 2. Run inference on a single fundus image
python scripts/infer.py \
    --image samples/example_fundus.jpg \
    --checkpoint weights/cd_former_drishti_seed43.ckpt \
    --output samples/example_output.png

# 3. Full evaluation on DRISHTI-GS
bash scripts/evaluate.sh drishti
```

---

## Repository structure

```
cd-former/
├── src/cd_former/             core package
│   ├── models/                DAVE, MSFL, C2QD blocks
│   ├── data.py                Fundus dataset + augmentation
│   ├── losses.py              Compound segmentation loss
│   ├── postproc.py            Geometry-Consistent vCDR Refinement (GCvR)
│   ├── evaluator.py           OD/OC Dice + AUC + vCDR MAE
│   └── trainer.py             Training loop
├── scripts/                   train, evaluate, infer, ensemble entry points
├── slurm/                     SLURM job templates for HPC training
├── configs/                   dataset and experiment YAMLs
├── tests/                     pytest smoke + unit tests
├── docs/
│   ├── MODEL_ZOO.md           download links and MD5 sums
│   ├── DATASETS.md            how to obtain each public dataset
│   └── TRAINING_RECIPE.md   step-by-step recipe to retrain the models
└── samples/                   one example image to test inference end-to-end
```

---

## Pre-trained models

Six RETFound-encoded CD-Former checkpoints are released, three independent seeds per dataset. See [`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md) for download links and per-checkpoint evaluation metrics.

| Checkpoint | Dataset | Seed | OD Dice | OC Dice | AUC | vCDR MAE | Size |
|---|---|---:|---:|---:|---:|---:|---:|
| `cd_former_drishti_seed42.ckpt` | DRISHTI-GS | 42 | 96.41 | 90.16 | 0.990 | 0.052 | ~1.3 GB |
| `cd_former_drishti_seed43.ckpt` | DRISHTI-GS | 43 | 96.75 | 90.57 | 0.990 | 0.062 | ~1.3 GB |
| `cd_former_drishti_seed44.ckpt` | DRISHTI-GS | 44 | 96.23 | 88.14 | 0.990 | 0.063 | ~1.3 GB |
| `cd_former_rimone_seed42.ckpt`  | RIM-ONE v3 | 42 | 95.19 | 75.10 | 0.947 | 0.076 | ~1.3 GB |
| `cd_former_rimone_seed43.ckpt`  | RIM-ONE v3 | 43 | 94.72 | 70.91 | 0.947 | 0.093 | ~1.3 GB |
| `cd_former_rimone_seed44.ckpt`  | RIM-ONE v3 | 44 | 94.82 | 71.59 | 0.961 | 0.087 | ~1.3 GB |

Weights are hosted on OneDrive; total ~8 GB. The download script verifies MD5 sums.

---

## Training

```bash
# 1. Prepare a dataset (DRISHTI-GS shown; same flow for RIM-ONE / REFUGE2 / SMDG-19)
python scripts/prepare_drishti_gs.py --source-dir /path/to/Drishti-GS1_files
python scripts/preprocess.py --config-name=data/drishti_gs

# 2. Single-seed training (~3 min on A100, 40 epochs)
python scripts/train.py \
    --data-root data/drishti_gs \
    --epochs 40 --seed 42

# 3. SLURM multi-seed sweep (3 seeds)
for s in 42 43 44; do sbatch slurm/train.slurm $s 40; done
```

See [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md) for the full end-to-end training recipe.

---

## Citation

Citation block will be filled in once the paper is accepted.

## License

This repository is currently private. License terms will be set at the public release.

