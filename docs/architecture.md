# Architecture

## Pipeline overview

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

## Components

### Backbone: ResNet-34 + DAC

`src/glaucoma_rdcnn/models/backbone.py` and `dac.py`.

ResNet-34 is taken from `torchvision.models`. The final classification head is removed; the four ResNet stages produce a feature tensor at 1/32 of the input resolution. A **Dense Atrous Convolution (DAC)** block is appended; its four parallel branches use dilation rates 1/3/6/12 and are summed (densely, like the original CE-Net DAC) before a 1x1 projection to `256` channels.

### Disc Proposal Network (DPN)

`src/glaucoma_rdcnn/models/dpn.py`.

Built on `torchvision.models.detection.rpn.RegionProposalNetwork` and `torchvision.models.detection.roi_heads.RoIHeads`. Anchors are sized for the OD scale (paper notes the OD spans roughly the central 30–40% of the cropped ROI). The classifier head is binary (background vs. disc).

### Disc Attention Module

`src/glaucoma_rdcnn/models/attention.py`.

Given the predicted OD bounding box, we rasterize a soft mask at the feature-map resolution, Gaussian-blur it for smoothness, and use it to gate the input feature in two ways: a spatial gate (1x1 conv from the soft mask to per-channel weights) and a channel gate (squeeze-and-excitation). The gated feature is fused with the original via a residual connection. This satisfies the paper's requirement that the cup head sees an OD-conditioned feature, since OC is always inside OD.

### Cup Proposal Network (CPN)

`src/glaucoma_rdcnn/models/cpn.py`.

Same architecture as DPN, smaller anchors. Operates on attended features.

### Geometry / inscribed ellipse

`src/glaucoma_rdcnn/geometry.py`.

For each predicted axis-aligned bbox `(x0, y0, x1, y1)`, we compute the largest inscribed ellipse with horizontal semi-axis `(x1-x0)/2` and vertical semi-axis `(y1-y0)/2`, then rasterize it onto a per-image mask. CDR follows directly: `vertical_diameter(OC) / vertical_diameter(OD)`.

## What's PAPER-UNSPEC

Marked with `# PAPER-UNSPEC:` comments in code. Summary:

- **DAC dilation rates** — the paper references DAC but does not list rates; we use `[1, 3, 6, 12]` (standard CE-Net schedule).
- **RPN anchor sizes / aspect ratios** — using Faster R-CNN-convention defaults adapted to the OD scale.
- **Optimizer schedule beyond "SGD with momentum"** — using cosine annealing with a 2-epoch warmup.
- **Train/val/test splits for RIM-ONE v3** — using a deterministic md5-keyed 70/15/15 since RIM-ONE has no canonical split.
- **OD localizer** — the paper cites Zou et al. 2018; we substitute a fast intensity-based heuristic that's good enough for ROI cropping. When ground-truth OD masks are available during preprocessing we use the GT centroid instead.

## What's NOT here yet

- Multi-scale feature pyramids (the paper uses a single-level feature; we follow that).
- Mask R-CNN-style instance masks (paper uses ellipse fitting from boxes).
- Snapshot ensembling for evaluation.
