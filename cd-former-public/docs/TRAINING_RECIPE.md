# CD-Former — full training recipe

Step-by-step recipe to retrain CD-Former from scratch on the four supported datasets and obtain the numbers reported in the paper from raw data + this repository.

## Hardware

- **GPU:** single NVIDIA A100 (40 GB or 80 GB). The training loop fits in 30 GB with the default `batch_size=16`.
- **CPU:** 8 cores recommended for data-loading throughput.
- **Storage:** ~25 GB after preprocessing all four datasets.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

## 1. Data preparation

Either download the preprocessed archives from `docs/MODEL_ZOO.md` (recommended, ~3 GB total) **or** start from the original releases:

```bash
# Adapter — extract raw triples (image + OD mask + OC mask)
python scripts/prepare_drishti_gs.py --source-dir /path/to/Drishti-GS1_files
python scripts/prepare_rim_one.py    --source-dir /path/to/RIM-ONE_r3
python scripts/prepare_refuge2.py    --source-dir /path/to/REFUGE2
python scripts/prepare_smdg19.py     --source-dir /path/to/SMDG-19

# Preprocess — CLAHE + 800x800 ROI crop
python scripts/preprocess.py --config-name=data/drishti_gs
python scripts/preprocess.py --config-name=data/rim_one_v3
python scripts/preprocess.py --config-name=data/refuge2
python scripts/preprocess.py --config-name=data/smdg19
```

## 2. RETFound encoder weights

CD-Former uses [RETFound MAE](https://github.com/rmaphoh/RETFound_MAE) (ViT-L/16, fundus-pretrained on 1.6 M images). Access to the HuggingFace checkpoint is gated; request at:

  https://huggingface.co/YukunZhou/RETFound_mae_natureCFP

Then set `HF_TOKEN` in your environment. The training script downloads the weights automatically on first run.

## 3. Training

3 seeds per dataset, 40 epochs (DRISHTI) or 60 epochs (others), ~3-5 min wall each on A100.

```bash
# Single seed (smoke-test)
python scripts/train.py --data-root data/drishti_gs --seed 42 --epochs 40

# Full 3-seed sweep on a SLURM cluster
for s in 42 43 44; do
    sbatch slurm/train.slurm drishti_gs $s 40
    sbatch slurm/train.slurm rim_one_v3 $s 60
done
```

## 4. Evaluation

```bash
# Per-checkpoint test-set evaluation
python scripts/evaluate.py \
    --checkpoint outputs/<run_name>/best.ckpt \
    --data-root data/drishti_gs

# Apply GCvR post-processing recipe sweep
python scripts/postprocess.py \
    --checkpoint outputs/<run_name>/best.ckpt \
    --data-root data/drishti_gs

# 3-seed ensemble
python scripts/ensemble.py \
    --checkpoints outputs/cd_former_drishti_seed42/best.ckpt \
                  outputs/cd_former_drishti_seed43/best.ckpt \
                  outputs/cd_former_drishti_seed44/best.ckpt \
    --data-root data/drishti_gs
```

## 5. (Optional) Qualitative visualisation of your own predictions

```bash
python scripts/make_qualitative_figure.py \
    --checkpoint outputs/cd_former_drishti_seed43/best.ckpt \
    --data-root data/drishti_gs \
    --top-k 4 \
    --output-pdf qualitative.pdf
```

## Expected numbers

After completing the above:

| Metric         | DRISHTI (best seed) | DRISHTI ensemble | RIM-ONE (best seed) | RIM-ONE ensemble |
|----------------|--------------------:|-----------------:|--------------------:|-----------------:|
| OD Dice        | 96.75               | 96.65            | 95.31               | 95.20            |
| OC Dice        | 90.57               | 90.23            | 75.36               | 73.77            |
| Glaucoma AUC   | 0.990               | 1.000            | 0.961               | 0.961            |
| vCDR MAE       | 0.052               | 0.054            | 0.076               | 0.084            |

Numbers may vary by ±0.5 pp Dice across runs due to CUDA non-determinism.
