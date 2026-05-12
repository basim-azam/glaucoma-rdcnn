"""Smoke test 2: Stage-1 ROI localizer (Circular-Hough + tiny U-Net).

Runs on CPU. Uses real DRISHTI fundus images if available, falls back to
the synthetic fundus fixture if data isn't on disk.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def _gt_centroid(mask_path: Path) -> tuple[float, float]:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(m > 127)
    return float(xs.mean()), float(ys.mean())


def test_hough_finds_disc_on_synthetic_fundus(fake_fundus, smoke_outputs) -> None:
    """The synthetic fundus has a bright disc at (540, 480) r=95. Hough should hit it."""
    from glaucoma_rdcnn_v2.models import hough_initial_guess

    pred = hough_initial_guess(fake_fundus)
    assert pred is not None, "Hough returned no circles on synthetic fundus"

    # Synthetic disc is at (cx=540, cy=480), tolerance ~30 px
    assert abs(pred.cx - 540) < 30, f"cx={pred.cx} too far from 540"
    assert abs(pred.cy - 480) < 30, f"cy={pred.cy} too far from 480"

    # Draw the prediction and GT for visual inspection
    overlay = fake_fundus.copy()
    cv2.circle(overlay, (int(pred.cx), int(pred.cy)), int(pred.r), (0, 255, 0), 3)
    cv2.circle(overlay, (540, 480), 95, (0, 0, 255), 2)
    cv2.imwrite(str(smoke_outputs / "test_02_synthetic.png"), overlay)


def test_hough_on_real_fundus(real_fundus_paths, smoke_outputs) -> None:
    """If real DRISHTI data is available, Hough should hit the disc on at least 4/5."""
    if not real_fundus_paths:
        import pytest

        pytest.skip("No real DRISHTI data — run scripts/prepare_drishti_gs.py first")

    from glaucoma_rdcnn_v2.models import hough_initial_guess
    from glaucoma_rdcnn_v2.models.roi_localizer import crop_roi

    hits = 0
    for img_path in real_fundus_paths:
        mask_path = img_path.parent.parent / "masks" / f"{img_path.stem}_od.png"
        if not mask_path.exists():
            continue

        img = cv2.imread(str(img_path))
        pred = hough_initial_guess(img)
        gt_cx, gt_cy = _gt_centroid(mask_path)

        if pred is not None:
            # Within 80px of GT centroid (~5% of image width)
            if abs(pred.cx - gt_cx) < 80 and abs(pred.cy - gt_cy) < 80:
                hits += 1

                # Verify ROI crop covers >95% of disc mask
                cropped_img, (x0, y0, x1, y1) = crop_roi(img, pred.cx, pred.cy, 512)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_in_crop = mask[max(0, y0):y1, max(0, x0):x1]
                total = (mask > 127).sum()
                kept = (mask_in_crop > 127).sum() if mask_in_crop.size else 0
                coverage = kept / total if total else 0
                assert coverage > 0.95, f"crop only covers {coverage:.2%} of disc on {img_path.name}"

            # Always save the overlay for inspection
            overlay = img.copy()
            cv2.circle(overlay, (int(pred.cx), int(pred.cy)), int(pred.r), (0, 255, 0), 3)
            cv2.circle(overlay, (int(gt_cx), int(gt_cy)), 5, (0, 0, 255), -1)
            cv2.imwrite(str(smoke_outputs / f"test_02_{img_path.stem}.png"), overlay)

    assert hits >= 4, f"Hough found disc on only {hits}/5 real fundus images"


def test_roi_unet_forward_shape() -> None:
    """The tiny U-Net produces (B, 3) output for (B, 3, 256, 256) input."""
    from glaucoma_rdcnn_v2.models import ROILocalizer

    net = ROILocalizer(in_channels=6, base_channels=16)  # smaller for smoke
    img = torch.randn(2, 3, 256, 256)
    out = net(img, hough_priors=[None, None])
    assert out.shape == (2, 3), f"expected (2, 3), got {out.shape}"
    assert torch.isfinite(out).all()
