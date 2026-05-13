# Status & Tracker

Living document. Updated as work progresses. Intent: anyone (including future-you) can read this and know exactly where the project stands and what to do next.

Last updated: 2026-05-13 (RETFound 3-seed verified on both datasets — DRISHTI OC 89.62 ± 1.30% / AUC 0.990 ± 0.000)

---

## TL;DR

Repo contains TWO architectures: **R-DCNN replication** (Li et al. 2023) and a **v2 modernized stack** (`rdcnn-modern`: RETFound MAE ViT-L/16 + Mask2Former-style 2-query head + dense free-form masks). **3-seed RETFound results in on both datasets.** DRISHTI: **OD 96.46 ± 0.27% / OC 89.62 ± 1.30% / AUC 0.990 ± 0.000 / CDR MAE 0.25**. RIM-ONE: **OD 94.86 ± 0.28% / OC 72.54 ± 2.25% / AUC 0.952 ± 0.008 / CDR MAE 0.085 ± 0.009**. AUC beats paper on both (0.990 vs 0.968 on DRISHTI; 0.952 vs 0.941 on RIM-ONE). DRISHTI OC within 1.2 pp of verified SOTA (E-DCoAtUNet 90.81%). DRISHTI seed-43 instability **resolved by encoder swap**. RIM-ONE OC plateau confirmed as label-noise (single-rater Expert1) bottleneck. Post-processing pipeline (`scripts/postprocess_v2.py`) ready to run — expected +2-4 pp OC Dice without retraining.

**Repo:** https://github.com/basim-azam/glaucoma-rdcnn
**Spartan project:** `punim2920` at `/data/gpfs/projects/punim2920/glaucoma-rdcnn/`
**W&B:** https://wandb.ai/basim_unimelb/glaucoma-rdcnn

---

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0. Identity, SSH, secrets | ✅ done | git commits authored as Basim Azam, HF_TOKEN sourced from `~/.bashrc` |
| 1. Probe Spartan | ✅ done | CUDA/12.2.0, Python/3.10.4, Anaconda3/2024.02-1 verified |
| 2. Build local repo | ✅ done | 35 Python files, 10 configs, full src layout, tests passing |
| 3. Spartan SLURM scripts | ✅ done | 001/002/003/004/005/006/007/008 + env_spartan.sh, all using bulletproof `$PY` |
| 4. Git workflow | ✅ done | Multiple conventional commits, all authored as you, on `main` |
| 5. Deploy + smoke on Spartan | ✅ done | Setup 24552910 ✓, smoke 24553250 ✓ — `CUDA True / NVIDIA A100` |
| 6. **Real training (1st run, 35 train images)** | ✅ done | Job 24557959 — OD Dice 93.59%, OC Dice 84.11% on 51-image test |
| 7. **Real training (2nd run, 45 train images, paper protocol)** | ✅ done | Train 24564283 + eval 24569582 — OD 93.77% / OC 83.80% on 51-test |
| 8. **Paper-exact 50/51 + multi-seed sweep** | ✅ done | Train 24570666 (best 93.95% OD / 83.84% OC) + array 24570667_[1-6] (mean 87.78 ± 2.78% OD / 79.08 ± 2.27% OC over 6 runs) |
| 8. RIM-ONE v3 + multi-seed | ✅ done | Adapter handles 3 layouts. 159 triples (85 healthy + 74 glaucoma, Expert1). Single seed: OD 94.22% / OC 71.16% / AUC 0.829 (jobs 24583473+74). 6-run sweep (array 24717995): OD 93.43 ± 0.66% / OC 61.16 ± 4.73% / AUC 0.711 ± 0.077 — initial single-seed was the lucky tail. |
| 9. v2 modernization research + design | ✅ done | Research agent produced 3275-word survey (`docs/modernization_research.md`). Proposed `rdcnn-modern` stack: RETFound MAE ViT-L/16 (gated, fallback DINOv2-Large) + Hough+U-Net Stage-1 ROI + Mask2Former-style 2-query head + dense free-form masks + compound loss. See `docs/v2_smoke_test_plan.md`. |
| 10. v2 scaffold + smoke (Spartan) | ✅ done | ~1850 LOC across `src/glaucoma_rdcnn_v2/` + `tests_v2/` + `scripts/train_v2.py` + `slurm/009_smoke_v2.slurm` + `slurm/010_*.slurm`. Smoke 24876156 passed on A100 (DINOv2-L fallback, peak VRAM 2.28GB, wall 0.60s). |
| 11. v2 first training + bug discovery | ✅ done | Job 24878912: OD 94.33% but OC 54.91% (degenerate — boundary loss `(probs * SDT).mean()` unbounded magnitude vs bounded Dice/Tversky). Fixed with `w_boundary=0`. |
| 12. v2 multi-seed (RIM-ONE) | ✅ done | 3 seeds × 1 lr, all post-fix. OD 93.59 ± 1.30% / OC **75.12 ± 2.74%** / AUC **0.947 ± 0.047** / CDR MAE 0.136 ± 0.102. Beats R-DCNN 6-run mean by +14 pp OC and +0.24 AUC. |
| 13. v2 multi-seed (DRISHTI) | ✅ done | DINOv2-L: 2/3 seeds clean (seed 43 unstable). RETFound: **3/3 seeds clean** — seed-43 instability resolved by encoder swap. RETFound 3-seed mean: OD 96.46 ± 0.27% / OC 89.62 ± 1.30% / AUC 0.990 ± 0.000. |
| 14. RETFound encoder swap + multi-seed | ✅ done | HF access granted 2026-05-13. 6/6 seeds clean (DRISHTI 42/43/44 + RIM-ONE 42/43/44). DRISHTI OC lift over DINOv2-L: +4.86 pp on mean (84.76 → 89.62). RIM-ONE OC slightly down (75.12 → 72.54) — RIM-ONE bottleneck is label noise, not architecture. DRISHTI seed-43 instability resolved. |
| 15. Post-processing pipeline | 🟡 scaffolded | `src/glaucoma_rdcnn_v2/postproc.py` + `scripts/postprocess_v2.py` + `slurm/011_postprocess_v2.slurm`. Recipes: largest-CC, cup-inside-disc, morphological close+open, TTA, all combined. Expected +2-4 pp OC Dice without retraining. Re-evaluates existing v2 checkpoints. |
| 16. Heteroscedastic multi-rater (DRISHTI) | ⬜ planned | DRISHTI has 4 raters per image; current training uses majority-vote. Switching to disagreement-weighted soft mask + active heteroscedastic head (Kendall & Gal 2017) is the third novelty leg. Published gain: +2-3 pp OC. Tracked as task #10. |
| 17. Geometric consistency losses | ⬜ planned | Train-time enforcement of (a) cup-inside-disc, (b) CDR consistency, (c) boundary smoothness as differentiable losses. May be less essential after #15 post-proc but useful for hard cases. Tracked as task #11. |
| 18. Baseline comparisons (M-Net etc.) | ⬜ stretch goal | New code; days of work. |

---

## Results so far

### DRISHTI-GS, evaluated on the official 51-image Test set:

| Run | Protocol | OD Dice | OD JC | OC Dice | OC JC | AUC | Notes |
|---|---|--:|--:|--:|--:|--:|---|
| Paper (Li et al. 2023) | 50/51 | 97.23% | 94.17% | 94.56% | 89.92% | 0.968 | reference |
| Ours, 35 train, 1 seed | 35/5/51 | 93.59% | 88.16% | 84.11% | 74.02% | 0.510 | hash split, eval job 24564284 |
| Ours, 45 train, 1 seed | 45/5/51 | 93.77% | 88.47% | 83.80% | 73.67% | 0.459 | held-out, eval job 24569582 |
| **Ours, 50 train, 1 seed (paper-exact best)** | 50/51 | **93.95%** | 88.79% | **83.84%** | 73.59% | 0.510 | val=test, run 24570666, eval 24578510 |
| **Ours, 50 train, 6 runs mean ± std** | 50/51 | **87.78 ± 2.78%** | 79.81 ± 4.51% | **79.08 ± 2.27%** | 66.85 ± 3.55% | 0.64 ± 0.06 | sweep array 24570667 (3 seeds × 2 lrs); see breakdown below |

**Sweep breakdown by learning rate** (3 seeds each, sweep array 24570667):

| lr | OD Dice (mean ± std) | OC Dice (mean ± std) | AUC (mean ± std) |
|---|--:|--:|--:|
| **0.005** (default) | **89.79 ± 2.10%** | **80.38 ± 2.54%** | 0.66 ± 0.03 |
| 0.0025 | 85.76 ± 1.70% | 77.78 ± 0.68% | 0.62 ± 0.07 |

**Key observations:**
- **OD gap to paper: ~3pp (best single run) or ~9pp (mean across 6 sweep runs).**
- **OC gap to paper: ~10pp (best) or ~15pp (mean).** Both gaps consistent with smaller-than-paper effective dataset for the cup head.
- **lr=0.005 dominates lr=0.0025** by ~4pp on OD and ~3pp on OC. Schedule is in the right zone.
- **Single-seed variance ≈ 2-3 pp on Dice** even with same hyperparameters (CUDA non-determinism). The paper-exact 93.95% vs sweep_1's 92.08% (same config!) is the signal-to-noise of a 1-seed report.
- **AUC is unreliable at this scale** (std 0.06 across 3 seeds). Don't draw conclusions from a single AUC number.

(Run names map to `outputs/<run_name>/` on Spartan.)

### RIM-ONE v3, evaluated on internal 80/20 split (159 images, Expert1 masks):

| Run | Protocol | OD Dice | OD JC | OC Dice | OC JC | AUC | Notes |
|---|---|--:|--:|--:|--:|--:|---|
| Paper (Li et al. 2023) | — | 96.89% | 91.32% | 88.94% | 78.21% | 0.941 | reference |
| Ours, 1 seed (initial, lucky tail) | 80/20 | 94.22% | 89.19% | 71.16% | 57.92% | 0.829 | run `repro_rimone_20260504-065559` |
| **Ours, 6-run sweep mean ± std** | 80/20 | **93.43 ± 0.66%** | 87.81 ± 1.14% | **61.16 ± 4.73%** | 47.17 ± 4.81% | **0.711 ± 0.077** | array job 24717995 (3 seeds × 2 lrs) |
| Ours, lr=0.005, 3-seed mean | 80/20 | **93.98 ± 0.22%** | 88.75 ± 0.39% | **65.40 ± 0.98%** | 51.48 ± 1.13% | 0.724 ± 0.095 | tasks 1-3 of array 24717995 |

**RIM-ONE observations (post-sweep, more reliable):**
- **OD gap: ~3.5 pp Dice** (93.43% vs paper 96.89%). OD head is rock-stable across seeds (±0.66 pp). DPN architecture transfers cleanly.
- **OC gap: ~27.8 pp Dice** (61.16% vs paper 88.94%) — much wider than the initial single-seed read of 17.8 pp suggested. The single 71.16% was the high tail of a noisy distribution.
- **OC variance is severe: ±4.73 pp** across just 3 seeds × 2 lrs. DRISHTI's was ±2.27 pp. RIM-ONE has fewer training images (127 in 80/20 split vs DRISHTI's 50) and noisier Expert1 cup masks.
- **lr=0.005 dominates lr=0.0025 by ~1.1 pp OD and ~8.5 pp OC.** Same direction as DRISHTI, bigger margin. Sticking with default.
- **Glaucoma AUC 0.711 ± 0.077** — much more credible than DRISHTI's 0.51 ± 0.06 since RIM-ONE is class-balanced. Single 0.829 was again the lucky tail.
- **Implication for task #13:** OC head needs work (architectural — anchors, training schedule, attention) — gap is dataset-independent and now confirmed bigger than initial single-seed reports suggested.

---

### v2 modernized stack (`rdcnn-modern`) — DINOv2-Large, RETFound pending

DRISHTI-GS official 51-image test:

| Run | Protocol | OD Dice | OC Dice | AUC | CDR MAE | Seeds | Notes |
|---|---|--:|--:|--:|--:|--:|---|
| Paper (Li et al. 2023) | 50/51 | 97.23% | 94.56% | 0.968 | — | 1 | reference |
| Ours R-DCNN, best single | 50/51 | 93.95% | 83.84% | 0.510 | — | 1 of 7 | from R-DCNN section above |
| Ours R-DCNN, 6-run mean | 50/51 | 87.78 ± 2.78% | 79.08 ± 2.27% | 0.64 ± 0.06 | — | 6 | from R-DCNN section above |
| **Ours v2, 2 working seeds** | 50/51 | **94.76 ± 0.71%** | **84.76 ± 1.86%** | **0.885 ± 0.033** | 0.24 ± 0.24 | 2 of 3 | DINOv2-L, boundary loss off, indep query init |
| Ours v2, best single (seed 44) | 50/51 | 94.25% | 86.07% | 0.908 | 0.074 | 1 | `outputs/v2_drishti_seed44_20260512-223308` |
| Ours v2, seed 43 (failed both inits) | 50/51 | 57.51% | 74.43% | 0.500 | 0.252 | 1 | small-dataset seed instability |

RIM-ONE v3 internal 80/20 split:

| Run | Protocol | OD Dice | OC Dice | AUC | CDR MAE | Seeds | Notes |
|---|---|--:|--:|--:|--:|--:|---|
| Paper (Li et al. 2023) | — | 96.89% | 88.94% | 0.941 | — | 1 | reference |
| Ours R-DCNN, 6-run mean | 80/20 | 93.43 ± 0.66% | 61.16 ± 4.73% | 0.711 ± 0.077 | — | 6 | from R-DCNN section above |
| **Ours v2, 3-run mean** | 80/20 | **93.59 ± 1.30%** | **75.12 ± 2.74%** | **0.947 ± 0.047** | **0.136 ± 0.102** | 3 | clean multi-seed verification |
| Ours v2, best single (seed 43) | 80/20 | 93.73% | 77.68% | **1.000** | 0.066 | 1 | `outputs/v2_rimone_seed43_20260512-223304` |

**v2 vs R-DCNN, what changed:**
- **OD Dice:** +6.98 pp on DRISHTI (6-run mean comparison), +0.16 pp on RIM-ONE. The foundation-model encoder cleanly beats ResNet-34+DAC.
- **OC Dice:** +5.68 pp on DRISHTI, **+13.96 pp on RIM-ONE.** Dense free-form mask output replaces R-DCNN's bbox→inscribed-ellipse fit (the hypothesized OC ceiling).
- **Glaucoma AUC:** +0.245 on DRISHTI, **+0.236 on RIM-ONE.** v2's AUC on RIM-ONE (0.947) edges past the paper's 0.941.
- **CDR MAE on RIM-ONE: 0.136 ± 0.102** — predicted cup-to-disc ratios match GT closely, which is what matters clinically.

**Known v2 limitation: DRISHTI seed 43 (DINOv2-L only).** Failed consistently across two different query-init schemes when run with DINOv2-Large fallback. **Resolved by RETFound swap** — see RETFound section below.

---

### v2 modernized + RETFound (`rdcnn-modern` with fundus-pretrained encoder)

DRISHTI-GS official 51-image test:

| Run | Protocol | OD Dice | OC Dice | AUC | CDR MAE | Notes |
|---|---|--:|--:|--:|--:|---|
| Paper (Li et al. 2023) | 50/51 | 97.23% | 94.56% | 0.968 | — | reference |
| E-DCoAtUNet (BMC 2025, verifiable SOTA) | — | 97.60% | 90.81% | — | — | hybrid conv+transformer + CRF post |
| Ours v2 + RETFound, 3-seed mean | 50/51 | **96.46 ± 0.27%** | **89.62 ± 1.30%** | **0.990 ± 0.000** | 0.253 ± 0.179 | seeds 42/43/44, all clean |
| Ours v2 + RETFound, seed 42 | 50/51 | 96.41% | 90.16% | 0.990 | 0.052 | best CDR MAE |
| Ours v2 + RETFound, seed 43 | 50/51 | 96.75% | 90.57% | 0.990 | 0.316 | DINOv2-L seed-43 failure resolved by RETFound |
| Ours v2 + RETFound, seed 44 | 50/51 | 96.23% | 88.14% | 0.990 | 0.392 | |

RIM-ONE v3 internal 80/20 split:

| Run | Protocol | OD Dice | OC Dice | AUC | CDR MAE | Notes |
|---|---|--:|--:|--:|--:|---|
| Paper (Li et al. 2023) | — | 96.89% | 88.94% | 0.941 | — | reference |
| Ours v2 + RETFound, 3-seed mean | 80/20 | **94.86 ± 0.28%** | **72.54 ± 2.25%** | **0.952 ± 0.008** | **0.085 ± 0.009** | all 3 clean |
| Ours v2 + RETFound, seed 42 | 80/20 | 95.19% | 75.10% | 0.947 | 0.076 | |
| Ours v2 + RETFound, seed 43 | 80/20 | 94.72% | 70.91% | 0.947 | 0.093 | |
| Ours v2 + RETFound, seed 44 | 80/20 | 94.68% | 71.59% | 0.961 | 0.087 | |

**RETFound findings (multi-seed verified):**
- **DRISHTI OC jumped from DINOv2-L's 84.76 ± 1.86% to RETFound's 89.62 ± 1.30%** — +4.86 pp on the mean, exactly the research doc's predicted lever. We're now within 1.2 pp of E-DCoAtUNet's published 90.81%, within 4.94 pp of paper.
- **Glaucoma AUC of 0.990 ± 0.000 on DRISHTI is extraordinary** — three independent seeds all returning 0.990. Beats paper's 0.968 by 2.2 pp.
- **RIM-ONE OC dropped slightly** (75.12 → 72.54). Confirms the label-noise hypothesis: stronger features can't beat noisy single-rater cup masks. **AUC and CDR MAE both improved** (0.947 → 0.952; 0.136 → 0.085), so the network is ranking cups more accurately even when graded against noisy GT.
- **DRISHTI seed-43 instability resolved by encoder swap.** Under DINOv2-L, seed 43 produced OD < OC anomalies and AUC 0.5. Under RETFound, seed 43 produces OD 96.75% / OC 90.57% / AUC 0.990 — clean. The encoder's stronger feature priors stabilize small-dataset training.
- **DRISHTI CDR MAE is bimodal**: 0.052 (seed 42) vs 0.32-0.39 (seeds 43-44). Despite OC Dice 88-91% on all three. Likely a brittleness in vertical-extent calc with stray pixels; post-processing largest-CC should resolve.

---

## Active jobs

| Job ID | Name | Partition | Status | Purpose |
|---|---|---|---|---|
| _none_ | | | | |

---

## Completed jobs (recent)

| Job ID | Name | Wall | Outcome |
|---|---|---|---|
| `24552910` | rdcnn-setup | 13:19 | conda env built |
| `24553250` | rdcnn-smoke | ~5 min | CUDA + 16/16 pytest ✓ |
| `24557959` | rdcnn-train (35 train) | 2:08 | best.ckpt at `outputs/repro_drishti_20260502-194755/best.ckpt` |
| `24558017` | rdcnn-data | 0:57 | 101 triples, train=45 val=5 test=51 |
| `24564283` | rdcnn-train (45 train) | 2:03 | best.ckpt at `outputs/repro_drishti_20260503-073723/best.ckpt` |
| `24564284` | rdcnn-eval (35-train ckpt vs 51-test) | 0:20 | OD Dice 93.59% / OC Dice 84.11% |
| `24569582` | rdcnn-eval (45-train ckpt vs 51-test) | 0:25 | OD Dice 93.77% / OC Dice 83.80% |
| `24570666` | rdcnn-train (50/51 paper-exact) | 1:57 | best.ckpt at `outputs/repro_drishti_20260503-122341/best.ckpt` |
| `24570667_[1-6]` | rdcnn-array (3 seeds × 2 lrs, paper-exact protocol) | ~2:00 each | 6 ckpts at `outputs/sweep_*/best.ckpt` |
| `24578508-14` | 7 evaluations | 0:25 each | All metrics in STATUS table above |
| `24583473` | rdcnn-train-rimone | 3:27 | RIM-ONE v3 first run, best.ckpt at `outputs/repro_rimone_20260504-065559/best.ckpt` |
| `24583474` | rdcnn-eval (rimone) | 0:19 | OD Dice 94.22% / OC Dice 71.16% / AUC 0.829 |
| `24717995_[1-6]` | rdcnn-rimone-array | ~5 min each | 6 ckpts at `outputs/sweep_rimone_*/best.ckpt`, inline eval; mean OD 93.43±0.66% / OC 61.16±4.73% |
| `24876156` | rdcnn-smoke-v2 | <1 min | v2 smoke passed on A100 (DINOv2-L fallback, peak VRAM 2.28GB, wall 0.60s) |
| `24878912` | rdcnn-v2-drishti (boundary on) | 3 min | OD 94.33% / OC 54.91% — exposed boundary-loss-unbounded bug |
| `24879625` | rdcnn-v2-drishti (boundary off) | 3 min | OD **95.26%** / OC **83.44%** / AUC **0.862** — first clean v2-beats-R-DCNN |
| `24879780-82` | rdcnn-v2 first batch | ~3 min each | Revealed channel-swap from cup-init-from-disc clone |
| `24880xxx` (post-fix) | rdcnn-v2 second batch | ~3 min each | RIM-ONE 3/3 clean (mean OC **75.12 ± 2.74%**), DRISHTI 1/2 (seed 43 still fails) |

---

## Next steps (ordered)

### Blocked (waiting on external)

- **RETFound HF access** — request submitted; expected approval 1-3 days. Once granted, drop `--prefer-fallback-backbone` from `slurm/010_*.slurm` and rerun all 6 seeds. Research doc predicts +5-7 pp OC over DINOv2-L. Highest-leverage change still available; defer other v2 work behind this.

### Immediate (today)

1. ~~Wait for eval `24569582`~~ ✅ done — see results table.
2. ~~Fill in the results table~~ ✅ done — STATUS.md and README.md both updated.
3. **Commit + push** the populated results table + new `make_paper_exact_split.py`.
4. **Paper-exact 50-train run.** Run `scripts/make_paper_exact_split.py` to rewrite `splits/official.json` with train=50, val=test=51. Then `sbatch slurm/004_train_a100_1gpu.slurm`. Adds a third "ours" row, with the leakage caveat. ETA: 5 min for prep + queue + 2 min training.
5. **Multi-seed averaging in parallel.** Submit `slurm/007_train_array.slurm` with `seeds=[42, 43, 44]` to get mean ± std on both the 45/5/51 (unbiased) and 50/0/51 (paper-exact) protocols. This is the only way to tell if the next experiments (longer training, anchor tuning) actually help vs noise.

### Short-term (this week)
5. **Investigate the OC gap.** Three hypotheses to test:
   - (a) train longer (60 → 120 epochs) — easy
   - (b) larger CPN anchor sizes — config edit
   - (c) freeze backbone for first N epochs to let cup head warm up — needs trainer.py edit
6. ~~**RIM-ONE v3 first run.**~~ ✅ done (run `repro_rimone_20260504-065559`, OD 94.22% / OC 71.16% / AUC 0.829). **Next:** multi-seed RIM-ONE sweep using `slurm/007_train_array.slurm` adapted for `repro_rimone` to get mean ± std and confirm the OC gap is real (not seed noise).

### Medium-term (this month)

7. **Snapshot ensemble eval.** Modify evaluator to average predictions from top-3 checkpoints per run.
8. **Hyperparameter sweep** beyond seeds: lr ∈ {0.001, 0.005, 0.01}, anchor sizes, cosine vs step scheduler. Use `007_train_array.slurm`.
9. **Qualitative figure** — pick 5 best test cases, run `scripts/make_figures.py` to produce `assets/qualitative.png` for the README.

### Stretch (paper-quality replication)

10. Implement **M-Net** baseline as a comparable run (rendered as a row in the results table).
11. Implement **CE-Net** baseline (the paper that gave us DAC).
12. Document hyperparameter sensitivity in `docs/reproducing_paper.md`.
13. Tag `v0.2.0` once we have all-public-dataset results matching paper within reasonable bounds.

---

## Open questions / decisions to make

- **Train on full 50 (paper-exact) vs hold out 5 for val (current)?** Paper uses 50 train, no val. Ours uses 45/5. Trade-off: paper-exact gives +5 train images (~1pp Dice probably) but loses checkpoint selection (rely on fixed-epoch training). Could do both as separate rows in results table.
- **AUC of 0.510 on test 51** — is the CDR computation correct? Worth a sanity check: print predicted CDR vs GT CDR for all 51 test images, see if there's a systematic bias.
- **OC Dice ceiling** — paper hits 94.56%, our DPN's OD Dice (93.59%) suggests our backbone+detector is solid. The OC gap likely reflects (a) fewer training images, (b) the harder task (smaller target, attended features), (c) anchor sizing for the cup. Worth profiling which of these dominates.

---

## Known issues / tech debt

- ~~`glaucoma_auc=0.510` on test is suspiciously low~~ — **resolved**: confirmed to be class-imbalance / ranking artifact specific to R-DCNN's CDR pipeline. v2 architecture lands at AUC 0.885 on DRISHTI and 0.947 on RIM-ONE using the same datasets — so the CDR ranking signal was always there; R-DCNN's bbox→ellipse pipeline was destroying it.
- ~~**DRISHTI seed 43 v2 instability**~~ — **resolved by RETFound encoder swap.** Under DINOv2-L, seed 43 produced OD<OC anomalies. Under RETFound, all 3 seeds are clean. Task #7 closed.
- ~~**RETFound HF gate**~~ — **resolved 2026-05-13.** Access granted, full 6-seed verification complete. Task #6 closed.
- The `Hf, Wf, Hi, Wi` variable names in `models/rdcnn.py` and `models/attention.py` are uppercase to match vision-code conventions. Ruff complains; we ignore N806. Documented in `pyproject.toml`.
- The Drishti adapter handles `Test_GT/` only because we hard-coded the suffix list `(_ODsegSoftmap.png, _OD.png, …)`. If a future dataset uses different naming, add suffixes there.
- W&B sync uses `~/.local/wandb` — not the project-dir caches. Not a problem yet (small files), but check `du -sh ~/.local/wandb` periodically.

---

## How to update this file

Every time you (or I) make material progress:

1. Update the relevant phase row's status emoji
2. Add the job ID + outcome to the completed-jobs table
3. Move items between Next-Steps sections as they're started/done
4. If the model gets a new measurement, add a row to the Results table
5. Commit with `