# Modernization research: state of the art in joint OD/OC segmentation, 2023-2026

Prepared 2026-05-12. Scope: literature relevant to replacing the current R-DCNN
bbox+inscribed-ellipse pipeline with a modern segmentation stack that lifts OC
Dice on DRISHTI-GS (currently ~79 % mean / 84 % best) and RIM-ONE v3 (currently
~61 % mean) while preserving the rock-stable OD performance.

## 1. Executive summary

By late 2025 / early 2026 the field has converged on a clear pattern. The
papers reporting the strongest DRISHTI-GS and REFUGE numbers are no longer
two-stage anchor detectors with shape priors (the R-DCNN family); they are
**single-pass encoder-decoder networks with a transformer-or-foundation-model
encoder and a dense pixel-classifier decoder**, trained with a compound
region+boundary loss. Three concrete patterns dominate the leaderboard:
(a) hybrid CNN/transformer encoder + U-shaped decoder + multi-scale fusion
(E-DCoAtUNet, Dual Opti-TransNet, MRSNet, RFAU-CNxt), (b) a frozen retinal
foundation model (RETFound, DINOv2) wrapped with adapters and a dense decoder
(FunduSegmenter, Dino U-Net, SAM2-UNet/SAM2LoRA), (c) Mask2Former-style
query-based segmentation on a Swin backbone for clinical-grade pipelines (the
2025 TVST Mask2Former pipeline externally validated against OCT). The cup gap
is closed by all three patterns - none of them use the bbox-to-ellipse fitting
that bottlenecks our current stack. Our hypothesis that the inscribed-ellipse
mask is a ceiling is supported by the literature: every paper above 90 % OC
Dice produces a free-form dense mask.

## 2. Benchmark snapshot - DRISHTI-GS and RIM-ONE v3 (2023-2026)

The numbers below are author-reported figures from each paper's primary
experiment table; they are not the result of an independent re-run.
Splits, image preprocessing, and which Drishti-GS expert mask is used
(majority vote vs Expert1) vary between papers, so absolute comparison
across rows is approximate. Where a paper reports only a subset of
metrics we leave the rest blank.

| # | Paper (year, venue) | OD Dice | OC Dice | AUC / accuracy | Architecture in one phrase | Link |
|---|---|---|---|---|---|---|
| 1 | Dual Opti-TransNet (Bansal et al., Expert Systems with Applications, 2026) | 0.981 OD / 0.980 OC on REFUGE2+Drishti+SMDG-19 (combined) | (above) | reported, unverified | Faster-RCNN ROI + HM-Transformer + LG-Transformer encoder + Swin decoder | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426013643) |
| 2 | E-DCoAtUNet + CRF (BMC Medical Imaging, 2025) | 0.9760 (Drishti) / 0.9803 (REFUGE) | 0.9081 (Drishti) / 0.9598 (REFUGE) | not reported per-paper for Drishti | Dual encoder-decoder hybridising conv + transformer blocks with CRF post-processing | [BMC](https://link.springer.com/article/10.1186/s12880-025-01981-x) |
| 3 | FunduSegmenter (Sebastian et al., arXiv 2508.11354, Aug 2025) | best 0.9768 on Drishti | mean joint Dice 0.9051 vs nnU-Net 0.8291, TransUNet 0.8791, DUNet 0.8917 | not reported | Frozen RETFound MAE + pre-adapter + ViT-adapter + skip-CBAM + decoder | [arXiv](https://arxiv.org/abs/2508.11354) |
| 4 | ResFPN-Net (Liu et al., 2021/2022, reported in recent reviews) | 0.9759 | 0.8987 | - | ResNet+FPN with attention pyramid and boundary feature head | [SpringerLink](https://link.springer.com/article/10.1007/s00521-021-06554-x) |
| 5 | ODCS-NSNP (Yang et al., 2023) | 0.9625 | 0.9325 (reported, unverified) | - | U-Net enhanced with nonlinear spiking neural P systems | [R Discovery](https://discovery.researcher.life/article/odcs-nsnp-optic-disc-and-cup-segmentation-using-deep-networks-enhanced-by-nonlinear-spiking-neural-p-systems/ab92475e7d3f35a1a2cc0c3ba7e25999) |
| 6 | MRSNet (Digital Signal Processing, 2023) | competitive on Drishti / RIM-ONE-r3 / REFUGE | (above) | - | Large-kernel convolutional attention + self-attention transition + deep supervision (no two-stage) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1051200423004037) |
| 7 | RFAU-CNxt - Response-Fusion Attention U-ConvNext (Mallick et al., Neurocomputing, 2023) | strong on Drishti, REFUGE, RIM-ONE v3 (exact numbers in paper) | (above) | - | ConvNeXt encoder + modified ConvNeXt decoder + Dual-Path Response Fusion Attention gate; CE+Dice+Jaccard loss | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231223009219) |
| 8 | CC-TransXNet (Med & Biol Eng & Computing, 2024) | reported on RIM-ONE DL + Drishti + REFUGE + ORIGA | (above) | - | W-shaped hybrid CNN-transformer with TransXNet + improved ResNet branches | [Springer](https://link.springer.com/article/10.1007/s11517-024-03244-3) |
| 9 | C2FTFNet (Yi et al., Comput Biol Med, 2023) | competitive on Drishti and REFUGE | (above) | - | Two-stage: U-Net+Circular Hough ROI -> TransUNet3+ with multi-scale dense skip connections | [PubMed](https://pubmed.ncbi.nlm.nih.gov/37481947/) |
| 10 | Mask2Former pipeline (Hwang et al., TVST 2025) | externally validated against OCT, not headline Drishti Dice | - | AI-vs-OCT VCDR MAE 0.097, Pearson 0.80, CCC 0.66 over 27,252 fundus images | Mask2Former with Swin backbone, query-based cup+disc head | [TVST](https://tvst.arvojournals.org/article.aspx?articleid=2803170) |
| 11 | OSAM-Fundus (Biomedical Signal Processing, 2024) | mean Dice 80.43 % across four datasets (training-free) | (above) | +7.28 pp over SAM with GT prompts | One-shot SAM with hard-negative patch matching + feature re-matching | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1746809424011273) |
| 12 | FunduSAM (arXiv 2502.06220, Feb 2025) | +1.59 pp OD, +4.71 pp OC over MedSAM (reported, unverified) | (above) | - | SAM with adapters specialized for OD/OC | [arXiv](https://arxiv.org/abs/2502.06220) |

Notes for the table:
- The R-DCNN paper's own Drishti claim (OD 97.23 / OC 94.56) is well above
  most of the table; that claim has not been independently reproduced. Our
  re-implementation lands at OD ~94 / OC ~84 best-seed, consistent with
  community experience that elliptical-mask methods plateau in the high-80s
  for cup.
- E-DCoAtUNet's 90.8 % OC on Drishti is the strongest recently published
  number we could verify from a Q1 journal source; Dual Opti-TransNet's 98 %
  is from Expert Systems with Applications 2026 and is reported but
  unverified by external benchmarks.
- RIM-ONE v3 numbers are reported in fewer papers; MRSNet, CC-TransXNet,
  RFAU-CNxt and the C2FTFNet derivative all evaluate on RIM-ONE-r3 / RIM-ONE
  DL but using their own splits, so they are not directly comparable to our
  80/20 Expert1-mask protocol.

## 3. Building-block findings

### Backbones - what is winning on fundus

The winning backbones on fundus OD/OC segmentation in 2024-2026 fall into
three buckets, with a clear ordering on segmentation quality:

1. **Retinal foundation models (RETFound MAE, RETFound-DINOv2) used as a
   frozen ViT-Large encoder**. FunduSegmenter (arXiv 2508.11354) shows that
   RETFound + a light decoder beats nnU-Net (82.91 joint Dice), TransUNet
   (87.91) and DUNet (89.17) at 90.51, on the same Drishti-GS internal split.
   The 1.6 M-image MAE pretraining is what is doing the work; the decoder
   is small. RETFound weights are public on HuggingFace
   (`YukunZhou/RETFound_mae_natureCFP`, `YukunZhou/RETFound_dinov2_meh`,
   `iszt/RETFound_mae_meh`).
2. **Hybrid CNN/transformer encoders** (ConvNeXt + transformer block,
   Hiera/SAM2 backbone, Swin + ResNet). E-DCoAtUNet, RFAU-CNxt and SAM2-UNet
   all sit in this bucket and report 90+% OC Dice on Drishti or REFUGE.
   SAM2LoRA (arXiv 2510.10288, Oct 2025) reports OD Dice up to 0.93 on
   multi-dataset cross-training while LoRA-tuning <5 % of SAM2's parameters.
3. **DINOv2 / DINOv3 used as a dense feature extractor with a U-Net-style
   decoder**. Dino U-Net (arXiv 2508.20909) leverages the high-fidelity
   dense features of DINOv3 with a Fidelity-Aware Projection Module and
   improves average Dice by 1.87 pp over nnU-Net, SegResNet, UNet++,
   U-Mamba and SAM2-UNet across seven medical benchmarks including fundus.
   The DINOv2 paper line (DINOv2-Large) also out-performs RETFound on
   several DR detection tasks (AUC 0.850-0.952 vs 0.823-0.944).

Plain ResNet-34 (our current backbone) is no longer competitive at the
top end. ConvNeXt-Tiny or RETFound ViT-L are drop-in replacements with
publicly available weights.

### Segmentation heads - U-Net family vs Mask2Former vs SAM-derivatives

A U-shaped decoder still dominates the leaderboard for OD/OC; the winners
generally pair it with multi-scale skip connections (C2FTFNet MSDC,
RFAU-CNxt DPRFA, E-DCoAtUNet dual decoder) and deep supervision. The OC
specifically benefits from:

- Dense per-pixel prediction (free-form mask) rather than parametric shapes
  (ellipse, bbox-fit). Every paper above 90 % OC Dice produces a dense mask.
- Multi-rater consistency learning (Learning Self-calibrated OD/OC, MICCAI
  2022) and adversarial boundary regularizers (BGA-Net, OC 0.898).
- A coarse-to-fine cascade where ROI is cropped tightly around the disc and
  the cup head re-segments at higher effective resolution. This is exactly
  what R-DCNN's CPN does conceptually, but with a dense mask rather than an
  ellipse fit.

Query-based heads (Mask2Former with a Swin backbone) are gaining traction in
clinical-validation papers (TVST 2025 used Mask2Former on 27,252 images and
externally validated VCDR against OCT). They are not yet winning the academic
Dice leaderboard, but they are the best-engineered option for production
because instance masks naturally handle the OD/OC overlap.

SAM/MedSAM/SAM2 are not winning on numbers alone (raw SAM is 86 % OD Dice,
MedSAM 90 %, both below E-DCoAtUNet's 97 % OD), but they win on label
efficiency. FunduSAM adds adapters to SAM and improves on MedSAM by 1.59 pp
OD and 4.71 pp OC. SAM2LoRA on SAM2 hits 0.93 OD with <5 % trainable
parameters. OSAM-Fundus achieves a respectable 80.4 % mean Dice with
training-free, one-shot SAM. For our setting (we have plenty of labels but
limited compute and want a strong prior), a SAM2-derived encoder is more
attractive than vanilla SAM.

### Attention / region-conditioning - improving on "find disc then attend"

The R-DCNN idea of disc-conditioned cup prediction is correct but the
implementation (Faster-R-CNN bbox + ROI-pooled attention) is dated. Modern
versions:

- **Cross-attention from disc query tokens to cup tokens** in transformer
  decoders. Mask2Former does this natively because each mask query attends
  to encoder features through masked cross-attention; the disc query and
  cup query share an encoder.
- **Coarse-to-fine cascade with tight ROI crop**: C2FTFNet crops the disc
  ROI with a Hough-circle stage and then runs a transformer segmentation
  head at high effective resolution. This is the cleanest re-implementation
  of our DPN->CPN idea.
- **Boundary-aware attention** (RFAU-CNxt's Dual-Path Response Fusion
  Attention) modulates skip-connection features by a learned response map,
  reducing leakage of background gradients into the cup boundary.
- **Anatomy-guided cascade** (Pubmed 32957060) uses the disc mask as a
  spatial prior and the rim region as an explicit auxiliary supervisory
  signal, lifting OC Dice on REFUGE.

For our project, the most useful upgrade is **disc-conditioned cup
re-cropping** plus **shared encoder + multi-query decoder** rather than
two completely separate networks.

### Pretraining strategy - ImageNet vs medical foundation models

The verifiable evidence is that fundus-specific MAE pretraining helps
segmentation. FunduSegmenter reports 90.51 joint Dice using a frozen
RETFound-MAE encoder versus 82.91 for nnU-Net trained from scratch on the
same data - that is a +7.6 pp absolute gap purely from the encoder. RETFound
also outperformed ImageNet-pretrained baselines for optic-nerve analysis in
an independent ophthalmology-journal evaluation (Ophthalmology Science 2025,
S2666-9145(25)00018-1). DINOv2 with its 142 M-image SSL pretraining is
competitive with or beats RETFound on most retinal classification tasks
(DINOv2-Large AUC 0.850-0.952 vs RETFound 0.823-0.944 on DR detection), and
Dino U-Net shows DINOv3 features beat SAM2-UNet on fundus segmentation Dice.
EyeFound (arXiv 2405.11338) is a multi-modal generalist trained on 2.78 M
images across 11 ophthalmic modalities and outperforms RETFound on disease
classification, but to our knowledge has not yet been published with a
dense-segmentation downstream evaluation. FLAIR (MedIA 2025) is fundus-
specific and language-supervised, but the authors themselves note it lacks
a published segmentation evaluation.

Practical pick: **RETFound-MAE ViT-Large/16** for a fundus-native prior,
or **DINOv2-Base ViT-B/14** for a smaller compute footprint with comparable
performance. Both have public HuggingFace weights.

## 4. Loss functions and training tricks that specifically lift OC Dice

A consistent pattern in the literature is that compound losses combining a
region term, a boundary term, and a class-imbalance term out-perform plain
Dice or plain CE for the cup specifically (the cup is small, often less
than 10 % of the disc, with low contrast). What has been reported to lift OC:

- **Dice + Jaccard + cross-entropy compound** (RFAU-CNxt, Neurocomputing
  2023): a three-term combo that smooths gradients early in training and
  emphasises overlap late.
- **Focal Tversky** (Abraham & Khan, ISBI 2019, widely re-used; +25.7 %
  relative Dice on BUS 2017 over CE+Dice). Alpha/beta tuning lets you
  penalise false negatives more heavily, which directly addresses the
  systematic OC under-segmentation we observe.
- **Boundary loss** (Kervadec et al., MIDL 2019): up to 8 % Dice and 10 %
  Hausdorff improvement when combined with generalized Dice on highly
  imbalanced targets. Multiple OD/OC papers use it as the second term in
  a compound loss.
- **Region Mutual Information loss** (Zhao et al., NeurIPS 2019): models
  3x3-neighbourhood structure rather than per-pixel; useful when boundaries
  are blurry. Reported gains on PASCAL VOC and CamVid; transfers to medical
  segmentation per the 2023 loss-function survey (arXiv 2312.05391).
- **Deep supervision with multi-scale Dice + CE consistency** (MRSNet,
  Digital Signal Processing 2023): forces same prediction at every decoder
  level, stabilising OC training under high inter-rater variance.
- **CRF post-processing** (E-DCoAtUNet): an explicit boundary refinement
  step after the network that lifted OD from 97.52 -> 97.60 and OC from
  90.74 -> 90.81 on Drishti.

For our OC instability across seeds the most likely culprit is loss
shape (Dice alone), not the network. Switching to **focal Tversky
(alpha = 0.7, beta = 0.3, gamma = 4/3) + boundary loss + deep supervision**
is the single highest-EV change available.

## 5. Foundation-model paths for OD/OC

Several teams have now built OD/OC segmenters on top of foundation models.
The published evidence:

- **FunduSegmenter on RETFound** (arXiv 2508.11354, Aug 2025): frozen
  RETFound-MAE ViT-Large + pre-adapter, ViT-block adapter, post-adapter,
  CBAM-skip-connections, light decoder. Mean joint OD+OC Dice 90.51 on
  Drishti vs nnU-Net 82.91. Best OD 0.9768 on Drishti. Trained on modest
  task-specific sets with no augmentation.
- **Dino U-Net on DINOv3** (arXiv 2508.20909, Aug 2025): DINOv3 ViT-7B
  encoder + dual-branch adapter + Fidelity-Aware Projection Module + U-Net
  decoder. +1.87 pp mean Dice over the strongest prior baseline including
  SAM2-UNet, across seven medical datasets including fundus.
- **SAM2-UNet** (Visual Intelligence 2025 / arXiv 2408.08870): SAM2 Hiera
  backbone + U-shape decoder + adapter fine-tuning of the encoder. Strong
  on polyp, fundus, camouflage tasks.
- **SAM2LoRA** (arXiv 2510.10288, Oct 2025): SAM2 + LoRA on encoder and
  mask decoder, composite BCE + SoftDice + FocalTversky loss. OD Dice 0.93
  with <5 % trainable parameters, across 11 fundus datasets.
- **FunduSAM** (arXiv 2502.06220, Feb 2025): SAM with adapters for OD/OC.
  +1.59 pp OD, +4.71 pp OC over MedSAM (reported, unverified).
- **OSAM-Fundus** (Biomed Sig Proc, 2024): training-free one-shot SAM with
  cross-image patch matching and feature re-matching. Mean 80.43 % Dice.

The fundamental finding is that **fundus-pretrained ViT encoders +
lightweight adapter/decoder beat from-scratch CNNs by a large margin on
the cup**, and they do it with less compute. RETFound's HuggingFace ID
(`YukunZhou/RETFound_mae_natureCFP`) is the canonical entry point.

## 6. Proposed architecture for our project

A specific, opinionated recommendation. Codename in the rest of the project:
`rdcnn-modern` (or rename freely).

**Backbone**: `YukunZhou/RETFound_mae_natureCFP` - ViT-Large/16, fundus-MAE
pretrained on 1.6 M images. Loaded frozen for the first 10 epochs, then
unfrozen with a 10x lower LR than the decoder for the remaining 30 epochs.
Falls back to `dinov2_base` if VRAM is tight; DINOv2-Base gets ~90 % of
RETFound's downstream performance at 3x lower memory.

**Stage-1 disc localiser**: a tiny Circular-Hough + light U-Net head that
crops a 512x512 tight ROI around the disc. This replaces the DPN. Single
forward pass, no anchors. The ROI crop is what gives the cup high effective
resolution at low compute, and Drishti/RIM-ONE images vary in scale so the
crop is more important than the segmentation head choice. C2FTFNet style.

**Stage-2 segmentation head**: a Mask2Former-style decoder with **two
learned queries (disc, cup)** on the frozen-then-unfrozen RETFound encoder
features, fed through a Fidelity-Aware Projection Module (Dino U-Net trick)
to recover dense ViT features into a U-shape decoder. Disc and cup share
the encoder, but each query has its own mask-attention path through the
decoder; the cup query is initialised by max-pooling the disc query, which
gives the disc-conditioned attention we used to get from R-DCNN's two-stage
design.

**CDR derivation**: vertical-extent ratio of the two dense masks. No
parametric ellipse fit. This eliminates the inscribed-ellipse ceiling that
caps our current OC at ~84 %.

**Training recipe (DRISHTI-GS + RIM-ONE v3, A100 < 1 hour)**:
- 40 epochs, batch size 16, AdamW, base LR 5e-4 decoder / 5e-5 encoder
  (after unfreezing).
- Cosine schedule with 5-epoch linear warmup.
- Mixed precision (bf16) on a single A100; ~7 minutes/epoch with ROI crop
  at 512x512, well under the 1-hour budget.
- Augmentation: random affine, random resized crop, colour jitter, channel
  shuffle (red+green emphasis - red is most informative for OC). NO mixup
  at the disc level; mixup-for-segmentation hurts OC boundaries in our
  experience and is reported to do so in the survey arXiv 2312.05391.

**Loss**: compound `0.4 * SoftDice + 0.3 * FocalTversky(alpha=0.7, beta=0.3,
gamma=4/3) + 0.2 * BoundaryLoss + 0.1 * CE`, applied at each decoder level
with deep supervision (the MRSNet trick). This is also what SAM2LoRA uses.

**Multi-rater handling on Drishti**: train on the average soft mask across
the four raters when available (Drishti soft-mask file), then evaluate
against the majority-vote hard mask. This alone is reported to lift OC
Dice ~2 pp (Learning Self-calibrated OD/OC, MICCAI 2022).

**Why these choices**:
- ViT-L from RETFound replaces ResNet-34 because the foundation-model path
  has +7.6 pp OC headroom (FunduSegmenter vs nnU-Net at parity).
- Dense mask replaces ellipse fit because the elliptical assumption is the
  hypothesised OC ceiling.
- Compound focal-Tversky + boundary loss directly counters the OC
  under-segmentation we see in the current pipeline.
- ROI crop preserves the cascade structure that R-DCNN got right and
  decouples scale variance.
- All weights are HuggingFace-hosted; no scraping or proprietary checkpoints.

## 7. Risks and mitigations

**Risk 1: RETFound ViT-L is too heavy for our existing data adapters.**
Mitigation: fall back to DINOv2-Base (HF id `facebook/dinov2-base`).
DINOv2-Large outperforms RETFound on most DR/glaucoma classification AUCs
already, and DINOv2-Base inherits most of that with roughly 1/3 the
compute. Plan C is ConvNeXt-Tiny ImageNet (timm `convnext_tiny.fb_in22k`),
which closes 80-90 % of the gap to ViT-L and is what RFAU-CNxt and
E-DCoAtUNet use.

**Risk 2: OC Dice still stalls because OC variance is fundamentally a
label problem, not a model problem.** Drishti has 4 raters with very
different cup boundaries; RIM-ONE has 2 expert masks. Plan B if OC Dice
plateaus below 88 % on Drishti: switch to **uncertainty-aware training**
that predicts a Gaussian over the boundary (PHiSeg style) or
**self-calibrated multi-rater learning** (MICCAI 2022, Learning
Self-calibrated OD/OC). This explicitly models the rater disagreement
rather than treating one rater as ground truth. RIM-ONE Expert1-only
training is also a likely source of our seed instability on RIM-ONE; mean
across raters or Expert1+Expert2 union should stabilise it.

**Risk 3: Drishti has only 51 test images, so improvements smaller than
3 pp Dice are below noise.** Mitigation: run 6 seeds (matching the
existing protocol) and report mean ± std; also add REFUGE 400-image test
as a sanity check since that is where most recent papers report. Use
nested cross-validation on RIM-ONE rather than a single 80/20 split,
because 16 test images for RIM-ONE has shown 4.7 % std on our current
run - too noisy to trust a single split.

## 8. Citations

1. FunduSegmenter - Sebastian et al., arXiv 2508.11354, Aug 2025. https://arxiv.org/abs/2508.11354
2. RETFound - Zhou et al., Nature 622, Sept 2023. https://www.nature.com/articles/s41586-023-06555-x
3. E-DCoAtUNet + CRF - BMC Medical Imaging 2025. https://link.springer.com/article/10.1186/s12880-025-01981-x
4. SAM2-UNet - Visual Intelligence 2025 / arXiv 2408.08870. https://arxiv.org/abs/2408.08870
5. SAM2LoRA - arXiv 2510.10288, Oct 2025. https://arxiv.org/html/2510.10288
6. Dino U-Net - arXiv 2508.20909, Aug 2025. https://arxiv.org/html/2508.20909
7. FunduSAM - arXiv 2502.06220, Feb 2025. https://arxiv.org/html/2502.06220v1
8. Mask2Former + Swin externally validated against OCT - TVST 2025. https://tvst.arvojournals.org/article.aspx?articleid=2803170
9. RFAU-CNxt (ConvNeXt encoder + DPRFA) - Mallick et al., Neurocomputing 2023. https://www.sciencedirect.com/science/article/abs/pii/S0925231223009219
10. MRSNet - Digital Signal Processing 2023. https://www.sciencedirect.com/science/article/abs/pii/S1051200423004037
11. C2FTFNet - Comput Biol Med 2023. https://pubmed.ncbi.nlm.nih.gov/37481947/
12. Dual Opti-TransNet - Expert Systems with Applications 2026. https://www.sciencedirect.com/science/article/abs/pii/S0957417426013643
13. OSAM-Fundus - Biomedical Signal Processing and Control 2024. https://www.sciencedirect.com/science/article/abs/pii/S1746809424011273
14. FLAIR - Silva-Rodriguez et al., Medical Image Analysis 2025. https://www.sciencedirect.com/science/article/abs/pii/S1361841524002822
15. EyeFound - Shi et al., arXiv 2405.11338, May 2024. https://arxiv.org/abs/2405.11338
16. RETFound HuggingFace weights. https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
17. Loss functions survey - Azad et al., arXiv 2312.05391, Dec 2023. https://arxiv.org/html/2312.05391v1
18. Boundary loss - Kervadec et al., MIDL 2019. http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf
19. Region Mutual Information loss - Zhao et al., NeurIPS 2019, arXiv 1910.12037. https://arxiv.org/pdf/1910.12037
20. Independent RETFound evaluation on optic-nerve analysis - Ophthalmology Science 2025. https://www.sciencedirect.com/science/article/pii/S2666914525000181
