# Status & Tracker

Living document. Updated as work progresses. Intent: anyone (including future-you) can read this and know exactly where the project stands and what to do next.

Last updated: 2026-05-03

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
| 7. **Real training (2nd run, 45 train images, paper protocol)** | ⏳ in progress | Train 24564283 done, eval 24569582 pending |
| 8. RIM-ONE v3 | ⬜ planned | Need to download the dataset + run same pipeline |
| 9. Multi-seed averaging | ⬜ planned | Use `slurm/007_train_array.slurm` |
| 10. README results table | ⬜ blocked on (7) | Will fill after eval2 |
| 11. Baseline comparisons (M-Net etc.) | ⬜ stretch goal | New code; days of work |

---

## Results so far

DRISHTI-GS, evaluated on the official 51-image Test set:

| Run | Train images | Val | OD Dice | OD JC | OC Dice | OC JC | AUC | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| Paper (Li et al. 2023) | 50 | — | 97.23% | 94.17% | 94.56% | 89.92% | 0.968 | reference |
| **Ours, 35 train, 1 seed** | 35 | 5 | **93.59%** | 88.16% | **84.11%** | 74.02% | 0.510 | hash split, w&b run `obqa7sqa` |
| **Ours, 45 train, 1 seed** | 45 | 5 | TBD | TBD | TBD | TBD | TBD | paper-protocol split, run `repro_drishti_20260503-073723` |

(Run names map to `outputs/<run_name>/` on Spartan.)

---

## Active jobs

| Job ID | Name | Partition | Status | Purpose |
|---|---|---|---|---|
| `24569582` | rdcnn-eval | gpu-a100-short | PD | eval 45-train ckpt against 51-test |

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

---

## Next steps (ordered)

### Immediate (today)

1. **Wait for eval `24569582` to complete.** Captures the 45-train-image checkpoint metrics on the 51-image test set. ETA: ~5 min once scheduled.
2. **Fill in the results table** in this file and in `README.md` with the eval2 numbers.
3. **Commit + push** the populated results table. This converts the README from "scaffold" to "first real numbers" status.

### Short-term (this week)

4. **Multi-seed averaging.** Submit `slurm/007_train_array.slurm` with 3 seeds × 1 lr. Updates `slurm/sweeps/repro.csv` to `seeds = [42, 43, 44], lr = [0.005]`. Outputs mean ± std for both heads. Usually +1-2 pp Dice from ensembling.
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
