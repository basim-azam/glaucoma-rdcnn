"""Run CD-Former on a single fundus image."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2, numpy as np, torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cd_former.models import build_backbone, build_seg_head
from cd_former.postproc import apply_recipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--recipe", default="cc+cid+morph")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(pretrained=False, image_size=cfg["encoder_size"],
                              prefer_fallback=cfg.get("prefer_fallback_backbone", False))
    head = build_seg_head(encoder_dim=backbone.feature_dim, decoder_dim=cfg["decoder_dim"],
                          num_decoder_layers=cfg["num_decoder_layers"], target_size=cfg["target_size"])
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])
    backbone.to(device).eval(); head.to(device).eval()

    img = cv2.cvtColor(cv2.imread(str(args.image)), cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (cfg["encoder_size"], cfg["encoder_size"]), interpolation=cv2.INTER_LINEAR)
    img_f = img_resized.astype(np.float32) / 255.0
    img_f = (img_f - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        feats = backbone(img_t)
        logits = head(feats.patch_features.float(), grid_hw=feats.grid_hw).mask_logits
    probs = torch.sigmoid(logits.float()).cpu().numpy()[0]
    disc, cup = apply_recipe(probs[0], probs[1], args.recipe)
    overlay = cv2.resize(img, (cfg["target_size"], cfg["target_size"]), interpolation=cv2.INTER_LINEAR)
    for mask, color in ((disc, (220, 30, 30)), (cup, (40, 180, 40))):
        if mask.any():
            cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnt, -1, color, 2)
    overlay = cv2.resize(overlay, (w0, h0), interpolation=cv2.INTER_LINEAR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    rows_d = np.where(disc.any(axis=1))[0]; rows_c = np.where(cup.any(axis=1))[0]
    h_d = rows_d[-1] - rows_d[0] + 1 if rows_d.size else 1
    h_c = rows_c[-1] - rows_c[0] + 1 if rows_c.size else 0
    cdr = h_c / h_d if h_d else 0.0
    print(f"vCDR: {cdr:.3f}  ({'glaucoma-suspect' if cdr > 0.5 else 'normal'})")
    print(f"wrote {args.output}")

if __name__ == "__main__":
    main()
