# Reproducing the R-DCNN paper

Steps to take this repo from a clean clone to numbers comparable to Table 1 of Li et al., *Eye* 2023.

## 1. Environment

Local CPU dev:

```bash
git clone https://github.com/basim-azam/glaucoma-rdcnn.git
cd glaucoma-rdcnn
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
pytest -q
```

For GPU on Spartan, see `spartan_guide.md`.

## 2. Data

```bash
bash scripts/download_data.sh all
python scripts/preprocess.py --config-name=data/drishti_gs
python scripts/preprocess.py --config-name=data/rim_one_v3
```

`download_data.sh` is idempotent. If a URL has rotted, the script prints the canonical landing page; download manually and place the zip in `data/<dataset>/` then re-run.

`preprocess.py` performs CLAHE on the L-channel, then crops an 800x800 ROI centered on the OD (using GT centroid where available, else the heuristic localizer). Output layout matches `docs/dataset_schema.md`.

## 3. Train

Single GPU (local or interactive Spartan session):

```bash
python scripts/train.py --config-name=experiment/repro_drishti training=single_gpu
```

Spartan, 1 A100:

```bash
sbatch slurm/004_train_a100_1gpu.slurm
```

Spartan, 4 A100 DDP:

```bash
sbatch slurm/005_train_a100_4gpu.slurm
```

## 4. Evaluate

```bash
python scripts/evaluate.py --config-name=experiment/repro_drishti +checkpoint=outputs/<run>/best.ckpt
```

Or on Spartan:

```bash
sbatch slurm/008_evaluate.slurm /data/gpfs/projects/punim2920/glaucoma-rdcnn/outputs/<run>/best.ckpt
```

## 5. Compare to paper

| Dataset      | Metric       | Paper  | Yours |
|--------------|--------------|-------:|------:|
| DRISHTI-GS   | OD Dice      | 97.23% |       |
|              | OD Jaccard   | 94.17% |       |
|              | OC Dice      | 94.56% |       |
|              | OC Jaccard   | 89.92% |       |
|              | Glaucoma AUC | 0.968  |       |
| RIM-ONE v3   | OD Dice      | 96.89% |       |
|              | OD Jaccard   | 91.32% |       |
|              | OC Dice      | 88.94% |       |
|              | OC Jaccard   | 78.21% |       |
|              | Glaucoma AUC | 0.941  |       |

## Tips

- Variability across seeds is real. Run the array job (`007_train_array.slurm`) for 3–6 seeds and report mean ± std.
- The DPN converges much faster than the CPN. Watch `cpn/loss_classifier` — if it plateaus before epoch ~10, your attention module may be over-suppressing the cup feature; widen the soft-mask sigma in `models/attention.py`.
- Data augmentation rotation by 90° is critical for RIM-ONE v3 (paper mentions this explicitly).
- For glaucoma AUC, the paper uses **CDR > 0.5** as the threshold. We compute AUC over CDR scores so the threshold doesn't change the AUC, but the threshold-derived sensitivity/specificity will.

## Reporting

Once you've trained and evaluated, fill in the "Yours" column above and the corresponding row in the README, then open a PR or commit directly. Don't claim reproduction until the numbers are within ~1 Dice / ~2 Jaccard percentage points of the paper.
