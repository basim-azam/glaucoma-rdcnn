# v2 (rdcnn-modern) smoke-test plan

**Status:** draft for review — written 2026-05-12. Approve this before scaffolding `src/glaucoma_rdcnn_v2/`.

**Goal.** Validate every new building block in the proposed `rdcnn-modern` stack (see `docs/modernization_research.md` §6) in isolation, on the cheapest possible hardware, before writing any training-loop code. Each test answers one question: "does this piece work, by itself, on real data?" If all five pass, the scaffold task is greenlit and we can build the full v2 package with high confidence. If any fail, we know exactly which sub-component needs a different approach (and we haven't sunk a week into integrating something that doesn't work).

**Scope.** Smoke only. No multi-epoch training, no metric matching, no checkpoint saving. The only metric we collect is "did it crash."

---

## The five tests

### Test 1 — RETFound checkpoint loads and produces features

**Question:** Can we load `YukunZhou/RETFound_mae_natureCFP` from HuggingFace and push a real DRISHTI fundus through it?

**File:** `tests_v2/test_01_backbone_loads.py`

**Setup:**
1. Set `HF_HOME` to the conda env's cache dir so weights persist between Spartan runs.
2. `from transformers import ViTModel` (or load via `timm` if RETFound is published in timm-compatible form — research agent's HF link suggests transformers-compatible).
3. Load real DRISHTI fundus from `data/drishti_gs/images/`, resize to 224×224, normalize.

**Assertions:**
- Model loads without error.
- Output feature tensor has shape `(1, 197, 1024)` for ViT-L/16 (196 patches + 1 CLS) at 224 input, OR the canonical RETFound feature shape — we'll resolve which on first run.
- All values are finite (no NaN/Inf).
- Parameter count is in the right ballpark: ~303M for ViT-L/16.

**Failure modes & responses:**
- **Gated checkpoint:** RETFound may require accepting an HF license. If `transformers` returns 403, the test prints the URL to accept the terms and we fall back to `facebook/dinov2-large` for the rest of the smoke tests.
- **Model architecture mismatch:** if the HF checkpoint isn't directly `ViTModel`-compatible (some MAE checkpoints need a separate ViTMAE loader), test prints the actual config and we adjust.
- **OOM on Spartan login node:** ViT-L is ~1.2GB in fp32. Should fit on the login node's memory but if not, the test reruns inside an interactive GPU session.

**Compute:** CPU is fine for the smoke (slow but works). Marked `@pytest.mark.slow`.

---

### Test 2 — Stage-1 ROI localizer produces a valid disc crop

**Question:** Can a Circular-Hough + tiny U-Net pipeline find the optic disc on a real DRISHTI fundus and crop a tight 512×512 ROI around it?

**File:** `tests_v2/test_02_roi_localizer.py`

**Setup:**
1. Load 5 real DRISHTI fundus images (we have 101 — pick the first 5 from `data/drishti_gs/images/`).
2. Load their corresponding OD masks from `data/drishti_gs/masks/` for ground-truth disc centers.
3. Run `cv2.HoughCircles` on the green channel after CLAHE (matches our existing preprocess) with a wide radius range.
4. Pick the highest-confidence circle that's roughly in the image's central 60%.
5. Crop a 512×512 ROI centered on the detected center.

**Assertions:**
- Hough returns at least one circle on all 5 images.
- The chosen circle's center is within 80 pixels (~5% of image width) of the ground-truth disc centroid.
- The 512×512 crop contains >95% of the GT disc mask pixels.
- Saves debug overlays to `tests_v2/_smoke_outputs/test_02/` for human inspection.

**Failure modes & responses:**
- **Hough misses on washed-out fundus:** documented failure mode. If >1 image fails, we add a light U-Net regression head (4-layer encoder predicting (cx, cy, r)) and the test re-runs with the U-Net guess as fallback. The research doc explicitly proposes this hybrid.
- **Crop clips disc:** the 512px box might cut off large discs on macro images. Test reports the actual disc-extent-vs-crop ratio per image.

**Compute:** CPU only. Pure-OpenCV test, ~1 second per image.

---

### Test 3 — Two-query Mask2Former-style head produces correct-shape masks

**Question:** Given ViT-L features, can a two-query decoder produce `(B, 2, H, W)` mask logits where the cup query attends to the disc-region features?

**File:** `tests_v2/test_03_seg_head.py`

**Setup:**
1. Implement a minimal Mask2Former-style head in `src/glaucoma_rdcnn_v2/models/mask2former_head.py`:
   - 2 learnable query embeddings (one OD, one OC).
   - 3 transformer decoder layers with cross-attention to the encoder features.
   - Cup query initialized as a max-pool of the disc query (the disc-attention modernization).
   - Mask-attention via dot product between query and pixel embeddings, upsampled to (H, W).
2. Feed it dummy ViT features `(B=2, N=1024, C=1024)` from a random tensor reshape (we don't need real RETFound here — test 1 already validates that).

**Assertions:**
- Output shape is `(B, 2, H_out, W_out)` with `H_out == W_out == 512` (matches ROI crop size).
- Outputs are finite.
- Gradient flows through both queries when we compute a fake loss and call `backward()`.
- The two output channels are not identical (queries actually disentangle).

**Failure modes & responses:**
- **Cup query initialization collapses both heads:** if outputs end up identical post-init, we switch to independent random init for the cup query and rely on training to learn the conditioning. Doc-style note added.
- **OOM at 512×512:** Mask2Former mask-attention can be memory-heavy. If we OOM on a smoke batch, we add a "small" config (256×256 attn maps, upsampled at the end) and proceed.

**Compute:** CPU. Random tensors, ~3 seconds.

---

### Test 4 — Compound loss is finite, scalar, and backward-able

**Question:** Does `0.4·SoftDice + 0.3·FocalTversky + 0.2·BoundaryLoss + 0.1·CE` produce a sensible scalar loss that decreases when predictions move toward targets?

**File:** `tests_v2/test_04_compound_loss.py`

**Setup:**
1. Implement the four losses in `src/glaucoma_rdcnn_v2/losses.py`:
   - SoftDice (multi-class, separate channels for OD and OC).
   - FocalTversky with α=0.7, β=0.3, γ=4/3.
   - BoundaryLoss using the signed-distance-transform formulation (precompute SDT once per target).
   - Cross-entropy.
2. Create three synthetic test cases on a 512×512 grid:
   - Random logits vs random target (should give finite, non-zero loss).
   - Perfect predictions (logits = +∞ where target == 1) → loss ≈ 0.
   - Nudged predictions (perfect + small noise) → loss < random-vs-random.

**Assertions:**
- All four loss components return finite scalars.
- Compound loss with random vs random is finite and > 0.
- Compound loss with near-perfect predictions is < 0.01 (sanity).
- `loss.backward()` succeeds.
- Gradients on input logits are non-zero everywhere (no dead channels).
- Loss with nudged predictions < loss with random predictions (monotonicity).

**Failure modes & responses:**
- **BoundaryLoss requires SDT precompute:** if `scipy.ndimage.distance_transform_edt` is missing from the env, test adds it via pip and re-runs.
- **FocalTversky γ=4/3 produces fractional power on negative values:** mathematically safe since (1 - TP/(TP+α·FN+β·FP)) ∈ [0,1], but test asserts.

**Compute:** CPU. ~5 seconds.

---

### Test 5 — End-to-end forward+backward on Spartan A100

**Question:** With everything wired together — RETFound encoder + ROI head + Mask2Former decoder + compound loss — does a single training step on a real DRISHTI batch complete without OOM, in bf16, on a real A100?

**File:** `tests_v2/test_05_e2e_smoke.py` + `slurm/009_smoke_v2.slurm`

**Setup:**
1. The pytest is the test logic. The SLURM script just runs `pytest tests_v2/test_05_e2e_smoke.py -v` on `gpu-a100-short` with a 10-minute time limit.
2. Loads a single DRISHTI batch of size 4 (real images + masks) from our existing data adapter.
3. Builds the full model: RETFound encoder → ROI crop → Mask2Former head.
4. Computes compound loss, calls `loss.backward()`, calls `optimizer.step()` once.

**Assertions:**
- Wall time < 30 seconds for the full step.
- Peak VRAM < 40GB (A100 has 80GB; we want headroom for batch 16 in training).
- Loss is finite before and after the step.
- After the step, `loss.item()` has changed (gradient actually applied).
- No `nan` or `inf` in gradients.
- bf16 path works (no precision-related errors).

**Failure modes & responses:**
- **OOM at batch 4:** reduce to batch 2; if still OOM, the architecture needs gradient checkpointing or a smaller backbone. Doc decision: fall back to DINOv2-Base per the research doc's Risk 1 mitigation.
- **NaN loss in bf16:** common with high-precision-sensitive losses (boundary, focal). Test catches this; falls back to fp32 with a config note that we lose ~30% of the speed budget.
- **HF_TOKEN issue on Spartan:** the SLURM script sources `HF_TOKEN` from `~/.bashrc` (already plumbed for our existing runs).

**Compute:** ~5 minutes on `gpu-a100-short`, single A100, one job slot.

---

## Sequencing

```
Test 1 (CPU) ──┐
Test 2 (CPU) ──┼── all 4 must pass ──> Test 5 (GPU/Spartan)
Test 3 (CPU) ──┤
Test 4 (CPU) ──┘
```

Tests 1-4 can run locally before any Spartan time is consumed. Test 5 only runs after 1-4 pass — it's our integration gate.

---

## Files to create (scaffolding minimum required to run smoke)

```
src/glaucoma_rdcnn_v2/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── backbone.py             # RETFound/DINOv2 loader (test 1)
│   ├── roi_localizer.py        # Hough + tiny U-Net (test 2)
│   └── mask2former_head.py     # 2-query decoder (test 3)
└── losses.py                   # SoftDice + FocalTversky + Boundary + CE (test 4)

tests_v2/
├── conftest.py                 # shared fixtures (fundus loader, etc.)
├── test_01_backbone_loads.py
├── test_02_roi_localizer.py
├── test_03_seg_head.py
├── test_04_compound_loss.py
└── test_05_e2e_smoke.py        # GPU-only, marked @pytest.mark.gpu

slurm/
└── 009_smoke_v2.slurm          # runs test_05 on gpu-a100-short

docs/
└── v2_smoke_test_plan.md       # this file
```

**Deliberately not built yet:** training loop, data augmentation pipeline, multi-rater handling, evaluator, configs, checkpointing, wandb integration. All of that comes after smoke passes and the scaffold task (#3) opens.

---

## How to run

```bash
# Local (CPU) — tests 1-4
cd C:\Users\basim\Downloads\glaucoma-rdcnn
pytest tests_v2/ -v -m "not gpu" --tb=short

# Spartan (GPU) — test 5
ssh spartan
cd /data/gpfs/projects/punim2920/glaucoma-rdcnn
sbatch slurm/009_smoke_v2.slurm
```

---

## Success criteria (the greenlight)

All five tests pass AND:
- Test 5's peak VRAM is < 40GB (so batch 16 at training time will fit comfortably).
- Test 5's wall time per step is < 5 seconds (extrapolates to <1 hour for 40 epochs × ~100 batches).
- RETFound loaded successfully (vs falling all the way back to ConvNeXt-Tiny).

If any of those fails: we still proceed, but the scaffold task starts with the fallback documented in the failure-mode notes above.

---

## What "smoke passing" unblocks

- Task #3 (scaffold `src/glaucoma_rdcnn_v2/`) — build the full package.
- Task #4 already covered by test 5 — close it.
- Task #5 (train v2 with multi-seed) becomes the next real chunk of work.

---

## Locked decisions (2026-05-12)

User greenlit "more time, more sophisticated architecture." Decisions:

1. **RETFound is the primary backbone, not a fallback.** We attempt the HF gated weights first; if blocked, we prompt the user to accept terms rather than silently downgrade. DINOv2-Large is a documented Plan B but not the default.
2. **Stage 1 is Hough + tiny U-Net hybrid from day 1**, not Hough-only. The U-Net is a 4-layer encoder predicting (cx, cy, log_r). Hough provides the prior; the U-Net refines. This matches C2FTFNet's published approach.
3. **Smoke test 5 runs on DRISHTI.** Exercises the multi-rater soft-mask path immediately so we don't discover loading issues 3 weeks into training.
4. **bf16 is the default**, fp32 is the documented fallback for the SDT-based boundary loss if NaN.

### Added: sophistication beyond the original plan

5. **Multi-rater uncertainty modeling from day 1** (was Plan B in the research doc, Risk 2). The seg head predicts both a mean mask and a per-pixel log-variance, trained with a heteroscedastic loss (Kendall & Gal 2017 formulation adapted for dense seg). At inference we still report the mean mask, but the variance is logged for downstream analysis. This is what the research doc identified as the most credible path to >88% OC Dice on DRISHTI given the 4-rater label noise.

6. **Cross-dataset joint training option.** The trainer supports concatenating DRISHTI + RIM-ONE with a small learnable dataset-conditioning token added to the encoder features. Off by default for the smoke tests; available as a config flag once we're past smoke. This is a stretch idea but the plumbing costs nothing to add now.
