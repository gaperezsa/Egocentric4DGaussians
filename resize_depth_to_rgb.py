"""
Resize a sequence's fused depth PNGs to match its RGB grid, in place.

Fixes the metric_depth_lab TARGET_WH bug for HOI4D: the delivered fused depth
is 475x265 but every method's HOI4D RGB/masks are the sequence's native grid
(e.g. Video1 = 468x262). Uses the SAME bilinear-on-metric-meters resample as
the lab's common/depth_io.resize_depth (PIL mode 'F'), so this is methodology-
consistent. 16-bit mm PNG in -> 16-bit mm PNG out at the RGB resolution.

Usage:
  python resize_depth_to_rgb.py --seq data/HOI4D/Video1
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="sequence dir, e.g. data/HOI4D/Video1")
    args = ap.parse_args()

    seq = Path(args.seq)
    depth_dir = seq / "depth"
    rgb = sorted(glob.glob(str(seq / "colmap" / "images" / "*.jpg")))
    assert rgb, f"no RGB in {seq}/colmap/images"
    out_w, out_h = Image.open(rgb[0]).size  # (W, H)

    pngs = sorted(glob.glob(str(depth_dir / "*.png")))
    d0 = np.array(Image.open(pngs[0]))
    print(f"[{seq}] {len(pngs)} depth PNGs {d0.shape[::-1]} (WxH) -> target {out_w}x{out_h}")
    if (d0.shape[1], d0.shape[0]) == (out_w, out_h):
        print(f"[{seq}] already matches RGB grid, nothing to do")
        return

    n = 0
    for p in pngs:
        d_mm = np.array(Image.open(p)).astype(np.float32)  # uint16 mm
        d_m = d_mm / 1000.0
        img = Image.fromarray(d_m, mode="F").resize((out_w, out_h), resample=Image.BILINEAR)
        d_m_r = np.asarray(img, dtype=np.float32)
        d_mm_r = np.clip(np.rint(d_m_r * 1000.0), 0, 65535).astype(np.uint16)
        Image.fromarray(d_mm_r, mode="I;16").save(p)
        n += 1
    # verify
    chk = np.array(Image.open(pngs[0]))
    print(f"[{seq}] resized {n} PNGs; now {chk.shape[::-1]} (WxH)")


if __name__ == "__main__":
    main()
