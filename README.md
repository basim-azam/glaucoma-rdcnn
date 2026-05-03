# glaucoma-rdcnn

Replication of **Region-based Deep CNN (R-DCNN)** for joint optic disc and optic cup segmentation, from
*Li, Xiang, Zhang et al., "Joint optic disk and cup segmentation for glaucoma screening using a region-based deep learning network", Eye 37, 1080–1087 (2023)* — DOI [10.1038/s41433-022-02055-w](https://doi.org/10.1038/s41433-022-02055-w).

R-DCNN reformulates joint optic-disc/optic-cup (OD/OC) segmentation as **object detection**: a Faster-R-CNN-style
network predicts an OD bounding box, a second head conditioned on the OD region predicts the OC bounding box,
and an inscribed ellipse is fit inside each box to produce the final segmentation. Cup-to-disc ratio (CDR)
follows directly from the two ellipses; CDR > 0.5 is the standard glaucoma-suspect threshold.

> **Status.** Scaffold + reference implementation. Numbers in the "this repo" column below are intentionally
> blank until full training is run on Spartan — see the open issues for the reproduction tracker.

## Architecture

```
RGB fundus  ──► CLAHE ──► OD localizer ──► 800x800 ROI crop
                                              │
                                              ▼
                       ┌────────────────────────────────────┐
                       │  ResNet-34 (ImageNet) + DAC block  │ ← shared backbone
                       └────────────────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
       ┌─────────────────┐    Disc Attention    ┌─────────────────┐
       │  DPN (RPN+ROI)  │ ─────────────────►  │  CPN (RPN+ROI)  │
       │  → OD bbox      │                     │  → OC bbox      │
       └─────────────────┘                     └─────────────────┘
                │                                   │
                └─────────────► inscribed ellipse fitting ◄────┘
                                          │
                                          ▼
                                 OD mask, OC mask, CDR
```

Components:

- **Backbone:** ResNet-34 (`torchvision`, ImageNet pretrained) with a Dense Atrous Convolution (DAC) block
  at the deepest stage to widen receptive field without losing resolution. See `src/glaucoma_rdcnn/models/dac.py`.
- **Disc Proposal Network (DPN):** anchor-based RPN + ROI head producing the OD bounding box. Built on
  `torchvision.models.detection.rpn` to avoid reimplementing the proposal machinery.
- **Cup Proposal Network (CPN):** same structure as DPN, but receives features that have been re-attended
  through the predicted OD region (Disc Attention Module).
- **Geometry:** the largest ellipse inscribed in each predicted axis-aligned box. CDR = vertical-cup-diameter / vertical-disc-diameter.

## Reproduction targets

Numbers from the paper (Table 1 + Fig 3) compared against this repo, all evaluated on the official 51-image DRISHTI-GS test set:

| Dataset    | Method                                     | Protocol  | OD Dice | OD JC  | OC Dice | OC JC  | Glaucoma AUC |
|------------|--------------------------------------------|-----------|--------:|-------:|--------:|-------:|-------------:|
| DRISHTI-GS | R-DCNN (Li et al. 2023)                    | 50/51     | 97.23%  | 94.17% | 94.56%  | 89.92% | 0.968        |
| DRISHTI-GS | R-DCNN (this repo, 45 train, 1 seed)       | 45/5/51 (held-out val) | 93.77%  | 88.47% | 83.80%  | 73.67% | 0.459        |
| DRISHTI-GS | R-DCNN (this repo, 50 train, **best of 7**) | 50/51 (val=test) | **93.95%** | 88.79% | **83.84%** | 73.59% | 0.510 |
| DRISHTI-GS | **R-DCNN (this repo, 50 train, 6-run mean)** | 50/51 | **87.78 ± 2.78%** | 79.81 ± 4.51% | **79.08 ± 2.27%** | 66.85 ± 3.55% | 0.64 ± 0.06 |
| DRISHTI-GS | R-DCNN (this repo, lr=0.005, 3-seed mean)  | 50/51 | 89.79 ± 2.10% | — | 80.38 ± 2.54% | — | 0.66 ± 0.03 |
| RIM-ONE v3 | R-DCNN (Li et al. 2023)                    | —         | 96.89%  | 91.32% | 88.94%  | 78.21% | 0.941        |
| RIM-ONE v3 | R-DCNN (this repo, 1 seed)                 | 80/20     | **94.22%** | 89.19% | **71.16%** | 57.92% | **0.829** |

**Notes on the gap:**
- OD head best single run is within ~3 pp Dice of the paper. The 6-run mean is ~9 pp behind, mostly because lr=0.0025 sweep runs underperform.
- OC head: ~10 pp gap (best) or ~15 pp (mean). The cup is the harder target with smaller surface area.
- **Single-seed variance is ~2-3 pp Dice** even with identical hyperparameters (CUDA non-determinism). A 1-seed report is genuinely uncertain at that scale.
- **lr=0.005 beats lr=0.0025** by ~4 pp OD / ~3 pp OC consistently. Default config is well-chosen.
- AUC has std 0.06 across seeds — single AUC numbers shouldn't be over-interpreted at this dataset size.
- "val=test" rows have a leakage caveat: best.ckpt is selected against the test set, matching the paper's effective protocol but slightly favouring the test number.
- **RIM-ONE v3:** OD gap is ~2.7 pp (similar magnitude to DRISHTI best), OC gap is ~17.8 pp (wider than DRISHTI). Glaucoma AUC 0.829 is much more credible here than on DRISHTI — RIM-ONE is class-balanced (85 healthy + 74 glaucoma).

See [`STATUS.md`](STATUS.md) for the live tracker, full run history, learning-rate breakdown, and next steps.

## Install

```bash
git clone https://github.com/basim-azam/glaucoma-rdcnn.git
cd glaucoma-rdcnn
python -m venv .venv && source .venv/bin/activate    # or use conda
pip install --upgrade pip
# CPU dev: torch from CPU index
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
pytest -q
```

For GPU on Spartan, see [`docs/spartan_guide.md`](docs/spartan_guide.md). The conda env is built by
`slurm/001_setup_env.slurm` at `/data/gpfs/projects/punim2920/glaucoma-rdcnn/conda_envs/glaucoma`.

## Datasets

Two public datasets are auto-downloaded; a third (in-house) is documented as a pluggable schema.

| Dataset       | Images | OC mask | Notes |
|---------------|-------:|:-------:|-------|
| DRISHTI-GS    | 101    | yes     | <https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php> |
| RIM-ONE v3    | 159    | yes     | Public mirror referenced in paper |
| Custom (in-house) | —  | yes     | Plug your data in via `configs/data/custom.yaml` — see `docs/dataset_schema.md` |

```bash
bash scripts/download_data.sh         # ~250 MB, idempotent, sha256-verified
python scripts/preprocess.py --config-name=data/drishti_gs
```

## Quickstart

```bash
# Smoke training (one batch, CPU)
python scripts/train.py +experiment=repro_drishti trainer.fast_dev_run=true

# Real training on a single GPU
python scripts/train.py +experiment=repro_drishti training=single_gpu

# Evaluate
python scripts/evaluate.py +experiment=repro_drishti checkpoint=outputs/best.ckpt

# Inference on a single fundus image
python scripts/infer.py --image=path/to/fundus.jpg --checkpoint=outputs/best.ckpt
```

## Spartan HPC

This repo is wired for the University of Melbourne **Spartan** cluster, project allocation `punim2920`.
The `slurm/` directory has scripts for env setup, data download, smoke test, single/multi-GPU training,
and evaluation. See [`docs/spartan_guide.md`](docs/spartan_guide.md).

```bash
ssh spartan
cd /data/gpfs/projects/punim2920
git clone https://github.com/basim-azam/glaucoma-rdcnn.git
cd glaucoma-rdcnn
sbatch slurm/001_setup_env.slurm    # one-time
sbatch slurm/003_smoke_test.slurm   # gpu-a100-short, ~30 min
sbatch slurm/004_train_a100_1gpu.slurm
```

## Citation

If you use this code, please cite both the original paper and this repository:

```bibtex
@article{li2023joint,
  title   = {Joint optic disk and cup segmentation for glaucoma screening using a region-based deep learning network},
  author  = {Li, Feng and Xiang, Wei and Zhang, Lin and Pan, Wenjuan and Zhang, Xuemin and Jiang, Mingshuai and Zou, Haidong},
  journal = {Eye},
  volume  = {37},
  pages   = {1080--1087},
  year    = {2023},
  doi     = {10.1038/s41433-022-02055-w}
}

@software{azam2026rdcnn,
  author = {Basim Azam},
  title  = {glaucoma-rdcnn: Replication of R-DCNN for joint OD/OC segmentation},
  year   = {2026},
  url    = {https://github.com/basim-azam/glaucoma-rdcnn}
}
```

## License

MIT — see [LICENSE](LICENSE).
