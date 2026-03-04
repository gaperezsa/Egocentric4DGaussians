"""
Visualize dynamic masks overlaid on RGB ground truth images (50/50 alpha blend).
Dynamic pixels are tinted red; static pixels show the original RGB.

Usage (from Egocentric4DGaussians/):
    python visualization_utils/visualize_dynamic_masks.py \
        --data_root data/ADT/cleanV2 \
        --out_dir /tmp/mask_vis/cleanV2 \
        --max_frames 50          # optional: limit number of frames
        --grid                   # optional: save a single grid image as well
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay_mask(rgb_img: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.5,
                 tint_color=(255, 0, 0)) -> np.ndarray:
    """Blend a boolean mask as a red tint over the RGB image."""
    out = rgb_img.copy().astype(np.float32)
    tint = np.zeros_like(out)
    tint[mask] = tint_color
    out = (1 - alpha) * out + alpha * tint
    # Where mask is False, keep original image unmodified
    out[~mask] = rgb_img[~mask].astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, len(text) * 7, 14], fill=(0, 0, 0))
    draw.text((2, 1), text, fill=(255, 255, 255))
    return np.array(pil)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="data/ADT/cleanV2",
                        help="Root of the dataset (contains colmap/images/ and dynamic_masks/)")
    parser.add_argument("--out_dir", type=str,
                        default=None,
                        help="Output directory (default: <data_root>/mask_vis)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit output to first N frames")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend alpha for mask overlay (0=no mask, 1=full red)")
    parser.add_argument("--grid", action="store_true",
                        help="Also write a single composite grid PNG")
    parser.add_argument("--grid_cols", type=int, default=8,
                        help="Columns in the grid image")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    image_dir = data_root / "colmap" / "images"
    mask_dir  = data_root / "dynamic_masks"
    out_dir   = Path(args.out_dir) if args.out_dir else data_root / "mask_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        sys.exit(f"[ERROR] Image directory not found: {image_dir}")
    if not mask_dir.exists():
        sys.exit(f"[ERROR] Mask directory not found: {mask_dir}")

    # Collect timestamps from mask files
    mask_files = sorted(mask_dir.glob("camera_dynamics_*.npy"))
    if not mask_files:
        sys.exit("[ERROR] No mask files found.")

    if args.max_frames:
        mask_files = mask_files[:args.max_frames]

    ts_re = re.compile(r"camera_dynamics_(\d+)\.npy")

    grid_frames = []
    missing_rgb = 0

    for i, mask_path in enumerate(mask_files):
        m = ts_re.match(mask_path.name)
        if not m:
            continue
        ts = m.group(1)

        rgb_path = image_dir / f"camera_rgb_{ts}.jpg"
        if not rgb_path.exists():
            # try png
            rgb_path = image_dir / f"camera_rgb_{ts}.png"
        if not rgb_path.exists():
            missing_rgb += 1
            continue

        mask = np.load(mask_path)           # (H, W) bool
        rgb  = np.array(Image.open(rgb_path).convert("RGB"))  # (H, W, 3)

        # Resize mask to match image if needed
        if mask.shape[:2] != rgb.shape[:2]:
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST)
            mask = np.array(mask_pil) > 127

        # Dynamic pixel count & percentage
        dyn_pct = mask.sum() / mask.size * 100

        vis = overlay_mask(rgb, mask, alpha=args.alpha)
        vis = add_label(vis, f"{ts} | dyn={dyn_pct:.1f}%")

        out_path = out_dir / f"vis_{ts}.jpg"
        Image.fromarray(vis).save(out_path, quality=95)

        if args.grid:
            grid_frames.append(vis)

        if (i + 1) % 20 == 0 or (i + 1) == len(mask_files):
            print(f"  [{i+1}/{len(mask_files)}] saved {out_path.name}  dyn={dyn_pct:.1f}%")

    print(f"\nDone. {len(mask_files) - missing_rgb} overlays saved to {out_dir}")
    if missing_rgb:
        print(f"  (skipped {missing_rgb} frames with missing RGB)")

    # ---- Optional grid ------------------------------------------------
    if args.grid and grid_frames:
        cols = args.grid_cols
        rows = (len(grid_frames) + cols - 1) // cols
        H, W = grid_frames[0].shape[:2]
        grid = np.zeros((rows * H, cols * W, 3), dtype=np.uint8)
        for idx, frame in enumerate(grid_frames):
            r, c = divmod(idx, cols)
            grid[r*H:(r+1)*H, c*W:(c+1)*W] = frame
        grid_path = out_dir / "grid.jpg"
        Image.fromarray(grid).save(grid_path, quality=90)
        print(f"  Grid saved → {grid_path}  ({cols}×{rows} = {len(grid_frames)} frames)")


if __name__ == "__main__":
    main()
