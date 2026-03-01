#!/usr/bin/env python3
"""
Extract deformed dynamic Gaussian centres as PLY pointclouds for every
*training* frame inside the dynamic phase of HOI4D Video1 (frames 56-139).

Training frames = even frame IDs  →  56, 58, 60, …, 138

Loading procedure mirrors train_dynamic_depth.py exactly:
  1. Build GaussianModel + Scene(dataset, gaussians, load_coarse=None)
     → reads colmap, create_from_pcd, set_aabb from initial sparse cloud,
       moves deformation network to CUDA
  2. torch.load(checkpoint)
  3. gaussians.restore(model_params, opt, stage_name)  ← same as training

Output:
  visualization_utils/dynamic_pointclouds_video1/frame_XXXXX.ply

Usage (from project root, with the training conda env active):
    python visualization_utils/extract_dynamic_pointclouds.py
"""

import os, sys

# ── project root on sys.path ──────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)          # needed so relative paths in cfg / data work

import torch
import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

# ── project imports ───────────────────────────────────────────────────────────
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from scene import Scene, GaussianModel


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration  – edit these three lines to change experiment / phase
# ══════════════════════════════════════════════════════════════════════════════

# Path to the .pth checkpoint (relative to project root)
CHECKPOINT  = "output/video2_time_smoothed/chkpnt_dynamics_depth_4100.pth"

# Data / model paths (must match the original training command)
SOURCE_PATH = "data/HOI4D/Video2/colmap"
MODEL_PATH  = "output/video2_time_smoothed"
CONFIGS     = "arguments/HOI4D/default.py"

# Phase of interest (inclusive); training frames = even IDs
PHASE_START = 182
PHASE_END   = 266

OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "dynamic_pointclouds_video2")

PHASE_START = PHASE_START if PHASE_START % 2 == 0 else PHASE_START + 1
# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def stage_from_checkpoint(ckpt_path: str) -> str:
    """Derive stage name from checkpoint filename, e.g. 'dynamics_depth'."""
    name = os.path.basename(ckpt_path)                    # chkpnt_dynamics_depth_4100.pth
    name = name.replace("chkpnt_", "").rsplit("_", 1)[0]  # dynamics_depth
    return name


def build_frame_time_map(colmap_sparse_dir: str) -> dict:
    """
    Replicate exactly what readColmapCameras() does:
      time = float(idx / len(cam_extrinsics))
    where idx is the enumeration index over the colmap binary dict
    (insertion order from the .bin file, NOT sorted by frame number).

    Returns {frame_id (int): time (float)}.
    """
    bin_path = os.path.join(colmap_sparse_dir, "images.bin")
    txt_path = os.path.join(colmap_sparse_dir, "images.txt")
    if os.path.exists(bin_path):
        from scene.colmap_loader import read_extrinsics_binary as _read
        cam_extrinsics = _read(bin_path)
    else:
        from scene.colmap_loader import read_extrinsics_text as _read
        cam_extrinsics = _read(txt_path)

    N = len(cam_extrinsics)
    frame_time = {}
    for idx, key in enumerate(cam_extrinsics):
        name = cam_extrinsics[key].name          # e.g. "camera_rgb_00056.jpg"
        base = os.path.splitext(os.path.basename(name))[0]   # "camera_rgb_00056"
        # Parse the numeric suffix (works for both "camera_rgb_00056" and "00056")
        fid  = int(base.split("_")[-1])
        frame_time[fid] = float(idx / N)
    return frame_time


def make_rainbow_colors(n: int, cmap_name: str = "gist_rainbow") -> np.ndarray:
    """
    Return (n, 3) uint8 array of sequential colors sampled from *cmap_name*.
    Gaussian i always gets the same color, so it can be tracked across frames.
    """
    import matplotlib.pyplot as plt
    cmap   = plt.get_cmap(cmap_name)
    t_vals = np.linspace(0.0, 1.0, n, endpoint=False)
    rgba   = cmap(t_vals)                        # (n, 4) float [0,1]
    return (rgba[:, :3] * 255).astype(np.uint8)  # (n, 3) uint8


def save_ply(path: str, xyz: np.ndarray, colors: np.ndarray) -> None:
    """
    Save an (N,3) float32 xyz array together with (N,3) uint8 RGB colors.
    Points are written in the order they appear in xyz/colors so that
    Gaussian i is always row i — enabling cross-frame color tracking.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dtype    = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"]     = xyz[:, 0]
    elements["y"]     = xyz[:, 1]
    elements["z"]     = xyz[:, 2]
    elements["red"]   = colors[:, 0]
    elements["green"] = colors[:, 1]
    elements["blue"]  = colors[:, 2]
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    print(f"    → saved {xyz.shape[0]:,} pts  {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stage = stage_from_checkpoint(CHECKPOINT)

    print(f"\n{'='*70}")
    print(f"  extract_dynamic_pointclouds.py")
    print(f"  checkpoint  : {CHECKPOINT}")
    print(f"  stage       : {stage}")
    print(f"  phase range : frames {PHASE_START}–{PHASE_END}  (even IDs only)")
    print(f"  output dir  : {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    # ── 1. Reconstruct argument namespaces exactly as training does ───────────
    #   Simulates: python train_dynamic_depth.py \
    #     --source_path <SOURCE_PATH> --model_path <MODEL_PATH> --eval
    parser  = ArgumentParser()
    model_p = ModelParams(parser)
    _       = PipelineParams(parser)
    opt_p   = OptimizationParams(parser)
    hyper_p = ModelHiddenParams(parser)

    sys.argv = [
        "extract_dynamic_pointclouds.py",
        "--source_path", SOURCE_PATH,
        "--model_path",  MODEL_PATH,
        "--eval",
    ]
    args = parser.parse_args()

    # Apply config-file overrides (same as --configs in the original run)
    if os.path.exists(CONFIGS):
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(CONFIGS)
        args   = merge_hparams(args, config)

    dataset = model_p.extract(args)
    opt     = opt_p.extract(args)
    hyper   = hyper_p.extract(args)

    # ── 2. Scene construction  (mirrors training) ─────────────────────────────
    #   - reads colmap sparse cloud
    #   - calls create_from_pcd  →  set_aabb from initial sparse cloud
    #   - moves _deformation to CUDA
    print("Building Scene from colmap data …")
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene     = Scene(dataset, gaussians, load_coarse=None)
    print(f"  dataset type : {scene.dataset_type}   maxtime : {scene.maxtime}")

    # ── 3. Load checkpoint and restore  (mirrors train_dynamic_depth.py) ──────
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model_params, first_iter = torch.load(CHECKPOINT, map_location="cpu")
    print(f"  saved at iteration : {first_iter}")

    gaussians.restore(model_params, opt, stage)
    # Training loads with no map_location so everything stays on CUDA.
    # We loaded with map_location="cpu", so tensors came back as CPU.
    # Push everything back to GPU now.
    gaussians._xyz             = gaussians._xyz.cuda()
    gaussians._dynamic_xyz     = gaussians._dynamic_xyz.cuda()
    gaussians._features_dc     = gaussians._features_dc.cuda()
    gaussians._features_rest   = gaussians._features_rest.cuda()
    gaussians._scaling         = gaussians._scaling.cuda()
    gaussians._rotation        = gaussians._rotation.cuda()
    gaussians._opacity         = gaussians._opacity.cuda()
    gaussians._deformation     = gaussians._deformation.cuda()

    n_total   = gaussians._xyz.shape[0]
    n_dynamic = int(gaussians._dynamic_xyz.sum().item())
    print(f"  Gaussians  total={n_total:,}   dynamic={n_dynamic:,}\n")

    if n_dynamic == 0:
        print("ERROR: no dynamic Gaussians – aborting.")
        sys.exit(1)

    # ── 4. Build frame→time map from actual colmap enumeration order ──────────
    colmap_sparse  = os.path.join(SOURCE_PATH, "sparse/0")
    frame_time_map = build_frame_time_map(colmap_sparse)

    # ── 5. Pre-select dynamic-Gaussian attributes ─────────────────────────────
    dyn_mask      = gaussians._dynamic_xyz
    means3D_dyn   = gaussians.get_xyz[dyn_mask]      # (M, 3)
    scales_dyn    = gaussians._scaling[dyn_mask]     # (M, 3)
    rotations_dyn = gaussians._rotation[dyn_mask]    # (M, 4)
    opacity_dyn   = gaussians._opacity[dyn_mask]     # (M, 1)
    shs_dyn       = gaussians.get_features[dyn_mask] # (M, 16, 3)
    M             = means3D_dyn.shape[0]

    # ── 6. Pre-compute per-Gaussian rainbow colors (fixed across all frames) ──
    # Gaussian i gets color i regardless of frame, so viewers can track each
    # point through the sequence by colour.
    colors = make_rainbow_colors(M)   # (M, 3) uint8
    print(f"Assigned rainbow palette ({M:,} unique colors from gist_rainbow)\n")

    # ── 7. Deformation loop ───────────────────────────────────────────────────
    frames = list(range(PHASE_START, PHASE_END + 1, 2))   # even → training ids
    print(f"Extracting {len(frames)} frames ({frames[0]} … {frames[-1]}) …\n")

    gaussians._deformation.eval()
    with torch.no_grad():
        for fid in frames:
            if fid not in frame_time_map:
                print(f"  frame {fid:3d}  WARNING: not in colmap, skipping")
                continue

            t           = frame_time_map[fid]
            time_tensor = torch.tensor(t, dtype=torch.float32).cuda().repeat(M, 1)

            means3D_deformed, *_ = gaussians._deformation(
                means3D_dyn,
                scales_dyn,
                rotations_dyn,
                opacity_dyn,
                shs_dyn,
                time_tensor,
            )

            pts      = means3D_deformed.detach().cpu().numpy()
            out_path = os.path.join(OUTPUT_DIR, f"frame_{fid:05d}.ply")
            print(f"  frame {fid:3d}  t={t:.5f}", end="")
            save_ply(out_path, pts, colors)

    print(f"\nDone – {len(frames)} PLY files in:\n  {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
