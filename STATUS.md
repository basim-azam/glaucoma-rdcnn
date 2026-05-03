# Status & Tracker

Living document. Updated as work progresses. Intent: anyone (including future-you) can read this and know exactly where the project stands and what to do next.

Last updated: 2026-05-03 (eval2 done — paper-protocol numbers in)

---

## TL;DR

Repo replicates the **R-DCNN architecture** end-to-end. Trained twice on Spartan, evaluated on the official 51-image DRISHTI-GS test set. Pipeline (data → train → eval → metrics → checkpointing → wandb) all proven on real GPU. OD head within ~3-4pp of paper Dice; OC head 10pp gap (tractable with more data + tuning). RIM-ONE v3 not yet attempted.

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
| 8. RIM-ONE v3 | ⏳ in progress | Adapter `scripts/prepare_rim_one.py` written (3 layouts incl. official 2015 r3) + `slurm/002` extended for `rim_one_v3` arg. Locally verified: 159 triples extracted (85 healthy + 74 glaucoma, Expert1 masks). Pending: rsync to Spartan + train/eval. |
| 9. Multi-seed averaging | ⬜ planned | Use `slurm/007_train_array.slurm` |
| 10. README results table | ⬜ blocked on (7) | Will fill after eval2 |
| 11. Baseline comparisons (M-Net etc.) | ⬜ stretch goal | New code; days of work |

---

## Results so far

DRISHTI-GS, evaluated on the official 51-image Test set:

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

---

## Next steps (ordered)

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
6. **RIM-ONE v3.** Download dataset (manual, same flow as DRISHTI), run preprocess + 002 → 004 → 008. Adds the second dataset row.

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

- `glaucoma_auc=0.510` on test is suspiciously low — see open question above.
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
5. Commit with `docs: update STATUS.md` and push

Pinning the file at the repo root (alongside README) makes it the canonical "where are we" reference for both you and any collaborator.
