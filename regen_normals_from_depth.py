"""
Regenerate surface-normal maps from the NEW fused depth, per sequence.

Reuses the exact back-projection + cross-product algorithm from
`data/adapt_adt_data.py` (compute_normals_from_depth_map / _batch), but:
  - reads intrinsics (fx, fy, cx, cy) from the sequence's COLMAP
    `colmap/sparse/0/cameras.txt` (PINHOLE), so it matches the actual
    depth/RGB resolution instead of the adapt script's assumed target size;
  - does NOT touch depth/ (unlike the full adapt pipeline, which overwrites it).

Writes:
  <seq>/normals/camera_normal_<id>.npy      (H, W, 3) float, camera-facing
  <seq>/normals_vis/camera_normal_<id>.png  (rotated 90deg CW for viewing)

Usage:
  python regen_normals_from_depth.py --seq data/ADT/cleanV2
  python regen_normals_from_depth.py --seq data/HOI4D/Video1
"""
import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


# --- verbatim from data/adapt_adt_data.py ---------------------------------
def compute_normals_from_depth_map(depth, fx, fy, cx, cy):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points_3d = np.stack([X, Y, Z], axis=-1)

    points_padded = np.pad(points_3d, ((1, 1), (1, 1), (0, 0)), mode='edge')
    t_u = points_padded[1:-1, 2:, :] - points_padded[1:-1, :-2, :]
    t_v = points_padded[2:, 1:-1, :] - points_padded[:-2, 1:-1, :]

    normals = np.cross(t_u, t_v, axisa=-1, axisb=-1, axisc=-1)
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm_mag + 1e-8)

    camera_direction = np.array([0.0, 0.0, -1.0])
    dot_product = np.sum(normals * camera_direction[np.newaxis, np.newaxis, :], axis=-1)
    flip_mask = dot_product > 0
    normals[flip_mask] *= -1
    return normals


def visualize_normals(normals):
    vis = ((normals + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
# --------------------------------------------------------------------------


def read_pinhole_intrinsics(cameras_txt: Path):
    """Parse a COLMAP PINHOLE cameras.txt -> (fx, fy, cx, cy)."""
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # CAMERA_ID MODEL W H fx fy cx cy
            model = parts[1]
            assert model in ("PINHOLE",), f"Unsupported camera model {model}"
            fx, fy, cx, cy = map(float, parts[4:8])
            return fx, fy, cx, cy
    raise RuntimeError(f"No camera line found in {cameras_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="path to sequence dir, e.g. data/ADT/cleanV2")
    args = ap.parse_args()

    seq = Path(args.seq)
    depth_dir = seq / "depth"
    normals_dir = seq / "normals"
    vis_dir = seq / "normals_vis"
    cameras_txt = seq / "colmap" / "sparse" / "0" / "cameras.txt"

    fx, fy, cx, cy = read_pinhole_intrinsics(cameras_txt)
    print(f"[{seq}] intrinsics fx={fx} fy={fy} cx={cx} cy={cy}")

    normals_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".png"))
    print(f"[{seq}] computing normals for {len(depth_files)} frames")

    # sanity: depth vs intrinsics resolution
    d0 = cv2.imread(str(depth_dir / depth_files[0]), cv2.IMREAD_UNCHANGED)
    print(f"[{seq}] depth shape {d0.shape} (H,W); cx*2={2*cx} cy*2={2*cy}")

    for fn in tqdm(depth_files, desc="normals"):
        depth_mm = cv2.imread(str(depth_dir / fn), cv2.IMREAD_UNCHANGED)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        normals = compute_normals_from_depth_map(depth_m, fx, fy, cx, cy)

        ts = fn.replace("camera_depth_", "").replace(".png", "")
        np.save(normals_dir / f"camera_normal_{ts}.npy", normals)
        vis = np.rot90(visualize_normals(normals), k=3)
        cv2.imwrite(str(vis_dir / f"camera_normal_{ts}.png"), vis)

    print(f"[{seq}] done -> {normals_dir}")


if __name__ == "__main__":
    main()
