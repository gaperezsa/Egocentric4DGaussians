#!/usr/bin/env python3
"""
Consolidated HOI4D Data Adaptation Pipeline

This script orchestrates the complete data adaptation workflow for HOI4D sequences:
  1. Decode RGB & depth videos to frames
  2. Process sparse depths and optional relative depth maps
  3. Generate metric depth maps with scale/shift optimization
  4. Convert color masks to binary masks
  5. Prepare COLMAP-compatible format

Data is automatically discovered from:
  - HOI4D_release/ : RGB video and relative depth maps
  - HOI4D_depth_video/ : Sparse GT depth video
  - HOI4D_annotations/ : Extrinsics (output.log), raw PCD, and masks
  - camera_params/ : Intrinsic camera parameters

Usage:
    python adapt_hoi4d_data.py \\
        --hoi4d_root /path/to/parent/folder \\
        --sequence_id ZY20210800001/H1/C8/N11/S321/s03/T2 \\
        --output /path/to/output \\
        [--relative_depth_folder /path/to/relative/depths] \\
        [--fps 15] \\
        [--epochs 5000]
"""

import argparse
import os
import subprocess
import shutil
from pathlib import Path
from copy import copy
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from natsort import natsorted
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import open3d as o3d
except ImportError:
    print("Warning: open3d not installed. PCD alignment will be skipped.")
    o3d = None


# ============================================================================
# PART 1: VIDEO DECODING
# ============================================================================

def decode_video(video_path: str, output_dir: str, fps: int):
    """Decode video to PNG frames using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        f'ffmpeg -i "{video_path}" '
        f'-f image2 -start_number 0 -vf fps=fps={fps} '
        f'-qscale:v 2 "{output_dir}/%05d.png" -loglevel quiet'
    )
    print(f"[VIDEO DECODE] {video_path} @ {fps}fps → {output_dir}")
    subprocess.run(cmd, shell=True, check=True)


# ============================================================================
# PART 2: DEPTH PROCESSING & METRIC DEPTH GENERATION
# ============================================================================

def load_sparse_depths(folder: str) -> List[np.ndarray]:
    """Load sparse depth PNGs from folder (in mm)."""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    arrs = []
    for f in files:
        im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Failed to read {f} from {folder}")
        arrs.append(im.astype(np.float32))  # in mm
    return arrs

def resize_depth_images(depth_dir: str, rgb_dir: str, output_dir: str):
    """Resize sparse depth frames to match RGB resolution."""
    os.makedirs(output_dir, exist_ok=True)
    samples = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg'))]
    assert samples, f"No RGB found in {rgb_dir}"
    w, h = Image.open(os.path.join(rgb_dir, samples[0])).size
    print(f"[DEPTH RESIZE] Resizing depth frames to {w}×{h}")
    for fn in sorted(os.listdir(depth_dir)):
        if not fn.endswith('.png'):
            continue
        im = Image.open(os.path.join(depth_dir, fn))
        im = im.resize((w, h), Image.NEAREST)
        im.save(os.path.join(output_dir, fn))


def resize_images_to_target(input_dir: str, output_dir: str, target_w: int, target_h: int):
    """Resize all images in a directory to the target dimensions."""
    os.makedirs(output_dir, exist_ok=True)
    for fn in sorted(os.listdir(input_dir)):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_dir, fn)
            img = Image.open(img_path)
            resized_img = img.resize((target_w, target_h), Image.NEAREST)
            resized_img.save(os.path.join(output_dir, fn))


def resize_relative_depths(input_folder: str, output_dir: str, target_w: int, target_h: int):
    """Resize relative depth .npy arrays to the target dimensions."""
    os.makedirs(output_dir, exist_ok=True)
    for fn in sorted(os.listdir(input_folder)):
        if fn.endswith('.npy'):
            arr = np.load(os.path.join(input_folder, fn))
            resized_arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            np.save(os.path.join(output_dir, fn), resized_arr)


def load_relative_depths(folder: str) -> List[np.ndarray]:
    """Load relative depth .npy files."""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy') and f.startswith('frame')])
    return [np.load(os.path.join(folder, f)) for f in files]


def optimize_scale_shift(X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, device) -> Tuple[float, float]:
    """Optimize scale (a) and shift (b) parameters: depth = a * relative + b"""
    print(f"[DEPTH OPT] Optimizing scale/shift over {len(X)} points...")
    X_t = torch.from_numpy(X).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)
    
    a = nn.Parameter(torch.tensor([1.0], device=device))
    b = nn.Parameter(torch.tensor([0.0], device=device))
    opt = optim.Adam([a, b], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100)
    
    for ep in range(epochs):
        opt.zero_grad()
        pred = a * X_t + b
        loss = torch.mean((pred - Y_t) ** 2)
        loss.backward()
        opt.step()
        scheduler.step(loss)
        if ep % 1000 == 0:
            print(f"  ep {ep:5d} | lr={opt.param_groups[0]['lr']:.6f} | loss={loss.item():.6f}")
    
    return a.item(), b.item()


def apply_and_save_metric_depths(relative_list: List[np.ndarray], a: float, b: float, out_folder: str):
    """Apply learned transform and save metric depth maps."""
    os.makedirs(out_folder, exist_ok=True)
    print(f"[DEPTH SAVE] Saving {len(relative_list)} metric depth maps...")
    for i, rel in enumerate(relative_list):
        dense = a * rel + b
        dense = np.clip(dense, 0, None)
        out = os.path.join(out_folder, f"camera_depth_{i:05d}.png")
        cv2.imwrite(out, dense.round().astype(np.uint16))


# ============================================================================
# PART 3: MASK PROCESSING
# ============================================================================

def mask_to_binary(image_path: str) -> np.ndarray:
    """Convert color PNG mask to binary (0/1)."""
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.uint8)
    return np.any(arr != 0, axis=-1).astype(np.uint8)


def get_reference_shape(ref_path: str) -> Tuple[int, int]:
    """Get (height, width) from reference image or folder."""
    if os.path.isfile(ref_path):
        img = Image.open(ref_path)
    else:
        imgs = sorted([
            f for f in os.listdir(ref_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not imgs:
            raise RuntimeError(f"No images found in {ref_path}")
        img = Image.open(os.path.join(ref_path, imgs[0]))
    w, h = img.size
    return h, w


def process_masks(masks_folder: str, target_ref: str, output_dir: str):
    """Convert color PNG masks to binary .npy, resized to target."""
    os.makedirs(output_dir, exist_ok=True)
    target_h, target_w = get_reference_shape(target_ref)
    
    masks = sorted([
        f for f in os.listdir(masks_folder)
        if f.lower().endswith('.png')
    ])
    if not masks:
        raise RuntimeError(f"No .png files found in {masks_folder}")
    
    print(f"[MASKS] Processing {len(masks)} masks → {target_w}×{target_h}")
    for i, fn in enumerate(masks):
        in_path = os.path.join(masks_folder, fn)
        mask = mask_to_binary(in_path)
        resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        out_fn = f"camera_dynamics_{i:05d}.npy"
        out_path = os.path.join(output_dir, out_fn)
        np.save(out_path, resized)


# ============================================================================
# PART 3.5: NORMAL MAP COMPUTATION FROM DEPTH 
# ============================================================================

def bilateral_filter_depth(depth: np.ndarray, sigma_spatial: float = 5.0, sigma_range: float = 0.1) -> np.ndarray:
    """Apply bilateral filter to depth map (edge-aware smoothing)."""
    # Convert to float32 if needed
    depth_f = depth.astype(np.float32)
    
    # Normalize depth for bilateral filter (should be ~0-1 or small range)
    depth_min, depth_max = np.nanmin(depth_f), np.nanmax(depth_f)
    if depth_max > depth_min:
        depth_norm = (depth_f - depth_min) / (depth_max - depth_min)
    else:
        return depth
    
    # Apply bilateral filter (8-bit or float conversion needed for cv2)
    depth_8bit = np.clip(depth_norm * 255, 0, 255).astype(np.uint8)
    filtered_8bit = cv2.bilateralFilter(depth_8bit, d=9, sigmaColor=sigma_range * 255, sigmaSpace=sigma_spatial)
    
    # Convert back to original range
    depth_filtered = (filtered_8bit.astype(np.float32) / 255.0) * (depth_max - depth_min) + depth_min
    
    return depth_filtered


def compute_depth_edge_confidence(depth: np.ndarray, rgb: Optional[np.ndarray] = None, 
                                   depth_threshold: float = 0.05, rgb_threshold: float = 0.1) -> np.ndarray:
    """Compute confidence mask based on depth and optional RGB gradients (edge-aware)."""
    # Compute depth gradient (Sobel)
    depth_f = depth.astype(np.float32)
    grad_x_d = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y_d = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_depth = np.sqrt(grad_x_d**2 + grad_y_d**2)
    
    # Normalize depth gradient
    grad_depth_norm = grad_depth / (np.max(grad_depth) + 1e-8)
    
    # If RGB provided, also consider RGB edges
    if rgb is not None and rgb.ndim == 3:
        rgb_gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x_r = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y_r = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_rgb = np.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_rgb_norm = grad_rgb / (np.max(grad_rgb) + 1e-8)
        grad_combined = np.maximum(grad_depth_norm, grad_rgb_norm)
    else:
        grad_combined = grad_depth_norm
    
    # Confidence: exponential decay at edges
    confidence = np.exp(-grad_combined / (depth_threshold + 1e-8))
    return confidence.astype(np.float32)


def backproject_depth_to_3d(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Back-project depth map to 3D points in camera space using intrinsics."""
    h, w = depth.shape
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    
    # Back-project: (u - cx) * D / fx, (v - cy) * D / fy, D
    Z = depth.astype(np.float32)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    # Stack into (H, W, 3) point cloud
    points_3d = np.stack([X, Y, Z], axis=-1)
    return points_3d


def compute_normals_from_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                               rgb: Optional[np.ndarray] = None, smooth_depth: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute normal maps from metric depth using A1 (back-project + cross-product).
    
    Args:
        depth: Metric depth map (H, W) in meters
        fx, fy, cx, cy: Camera intrinsics
        rgb: Optional RGB image for edge detection
        smooth_depth: Whether to apply bilateral smoothing first
    
    Returns:
        normals: (H, W, 3) unit normal vectors
        confidence: (H, W) edge confidence map
    """
    # Step 1: Smooth depth (edge-aware)
    if smooth_depth:
        depth_smooth = bilateral_filter_depth(depth, sigma_spatial=5.0, sigma_range=0.05)
    else:
        depth_smooth = depth.astype(np.float32)
    
    # Step 2: Back-project to 3D point cloud
    points_3d = backproject_depth_to_3d(depth_smooth, fx, fy, cx, cy)
    h, w = depth.shape
    
    # Step 3: Compute tangent vectors via central differences
    # Pad points with edge replication for boundary handling
    points_padded = np.pad(points_3d, ((1, 1), (1, 1), (0, 0)), mode='edge')
    
    # Horizontal tangent: X(u+1, v) - X(u-1, v)
    t_u = points_padded[1:-1, 2:, :] - points_padded[1:-1, :-2, :]  # (H, W, 3)
    
    # Vertical tangent: X(u, v+1) - X(u, v-1)
    t_v = points_padded[2:, 1:-1, :] - points_padded[:-2, 1:-1, :]  # (H, W, 3)
    
    # Step 4: Cross product → normal
    # n = t_u × t_v
    normals = np.cross(t_u, t_v, axisa=-1, axisb=-1, axisc=-1)  # (H, W, 3)
    
    # Step 5: Normalize to unit vectors
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm_mag + 1e-8)
    
    # Step 6: Orient to face camera (flip if n·[0,0,-1] > 0)
    camera_direction = np.array([0.0, 0.0, -1.0])  # Camera looks down -Z
    dot_product = np.sum(normals * camera_direction[np.newaxis, np.newaxis, :], axis=-1)
    flip_mask = dot_product > 0
    normals[flip_mask] *= -1
    
    # Step 7: Compute confidence from edge detection
    confidence = compute_depth_edge_confidence(depth, rgb=rgb, depth_threshold=0.05, rgb_threshold=0.1)
    
    return normals.astype(np.float32), confidence.astype(np.float32)


def visualize_normals(normals: np.ndarray) -> np.ndarray:
    """Convert normal map to RGB visualization (Red=+X, Green=+Y, Blue=+Z)."""
    # Normals are in [-1, 1], convert to [0, 255]
    vis = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    return vis


def process_normals_from_metric_depth(depth_folder: str, intrinsics_path: str, rgb_folder: Optional[str],
                                      output_normal_dir: str, output_vis_dir: str):
    """
    Batch process all depth frames to compute and save normal maps.
    
    Args:
        depth_folder: Folder with metric depth .png files
        intrinsics_path: Path to intrin.npy (3x3 camera matrix)
        rgb_folder: Optional folder with RGB frames for edge detection
        output_normal_dir: Where to save .npy normal maps
        output_vis_dir: Where to save .png normal visualizations
    """
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)
    
    # Load intrinsics
    K = np.load(intrinsics_path)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    print(f"[NORMALS] Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Get list of depth files
    depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])
    rgb_files = None
    if rgb_folder and os.path.exists(rgb_folder):
        rgb_files = {i: os.path.join(rgb_folder, f) for i, f in enumerate(sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith(('.png', '.jpg'))]))[:len(depth_files)]}
    
    print(f"[NORMALS] Processing {len(depth_files)} depth frames...")
    for i, depth_fn in enumerate(depth_files):
        # Load depth (uint16, mm) → convert to float32, meters
        depth_path = os.path.join(depth_folder, depth_fn)
        depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth_meters = depth_uint16 / 1000.0  # Convert mm to meters
        
        # Load RGB if available
        rgb = None
        if rgb_files and i in rgb_files:
            rgb = cv2.imread(rgb_files[i])
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  # BGR → RGB
        
        # Compute normals and confidence
        normals, confidence = compute_normals_from_depth(depth_meters, fx, fy, cx, cy, rgb=rgb, smooth_depth=True)
        
        # Save normal map (.npy)
        normal_fn = f"camera_normal_{i:05d}.npy"
        normal_path = os.path.join(output_normal_dir, normal_fn)
        np.save(normal_path, normals)
        
        # Save confidence map (.npy)
        conf_fn = f"camera_normal_confidence_{i:05d}.npy"
        conf_path = os.path.join(output_normal_dir, conf_fn)
        np.save(conf_path, confidence)
        
        # Visualize and save
        normals_vis = visualize_normals(normals)
        vis_fn = f"camera_normal_{i:05d}.png"
        vis_path = os.path.join(output_vis_dir, vis_fn)
        cv2.imwrite(vis_path, cv2.cvtColor(normals_vis, cv2.COLOR_RGB2BGR))
        
        if (i + 1) % 10 == 0 or i == len(depth_files) - 1:
            print(f"  [{i+1}/{len(depth_files)}] Processed {depth_fn} → normal range: [{normals.min():.3f}, {normals.max():.3f}]")
    
    print(f"[NORMALS] Saved {len(depth_files)} normal maps to {output_normal_dir}")
    print(f"[NORMALS] Saved {len(depth_files)} normal visualizations to {output_vis_dir}")


# ============================================================================
# PART 4: COLMAP ADAPTATION
# ============================================================================

def get_image_shape(path: str) -> Tuple[int, int]:
    """Get (height, width) from image/video file or folder."""
    if os.path.isfile(path):
        ext = path.lower()
        if ext.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(path)
            ret, f = cap.read()
            if not ret:
                raise RuntimeError(f"Cannot read {path}")
            h, w = f.shape[:2]
            cap.release()
            return h, w
        else:
            w, h = Image.open(path).size
            return h, w
    elif os.path.isdir(path):
        imgs = sorted(p for p in os.listdir(path) if p.lower().endswith(('.png', '.jpg')))
        if not imgs:
            raise RuntimeError(f"No images in {path}")
        w, h = Image.open(os.path.join(path, imgs[0])).size
        return h, w
    else:
        raise RuntimeError(f"{path} not found")


def load_trajectory_from_log(log_path: str):
    """Load camera poses from HOI4D output.log file."""
    if o3d is None:
        raise RuntimeError("open3d required for loading trajectory")
    traj = o3d.io.read_pinhole_camera_trajectory(log_path).parameters
    print(f"[COLMAP] Loaded {len(traj)} camera poses from {log_path}")
    return traj


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q


def write_cameras_txt(out_folder: str, K: np.ndarray, w: int, h: int):
    """Write COLMAP cameras.txt."""
    p = os.path.join(out_folder, "cameras.txt")
    with open(p, 'w') as f:
        f.write("# Camera list\n")
        f.write("# CAM_ID, MODEL, W, H, PARAMS\n")
        f.write(f"1 PINHOLE {w} {h} {K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n")
    print(f"[COLMAP] Wrote {p}")


def write_images_txt(out_folder: str, traj):
    """Write COLMAP images.txt."""
    p = os.path.join(out_folder, "images.txt")
    with open(p, 'w') as f:
        f.write("# Image list\n")
        f.write("# ID, QW, QX, QY, QZ, TX, TY, TZ, CAM_ID, NAME\n")
        for i, cam in enumerate(traj):
            P = copy(cam.extrinsic)
            R, t = P[:3, :3], P[:3, 3]
            qw, qx, qy, qz = rotmat2qvec(R)
            name = f"camera_rgb_{i:05d}.jpg"
            f.write(f"{i} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {name}\n\n")
    print(f"[COLMAP] Wrote {p}")


def write_points3d_ply(out_folder: str, pcd):
    """Write COLMAP points3D.ply from raw PCD."""
    pts = np.asarray(pcd.points)
    cnt = pts.shape[0]
    if cnt > 200000:
        pcd = pcd.random_down_sample(200000 / cnt)
        print(f"[COLMAP] Downsampled points {cnt} → {len(pcd.points)}")
    
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() \
        else np.zeros((len(pcd.points), 3), np.uint8)
    
    ply = os.path.join(out_folder, "points3D.ply")
    with open(ply, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pcd.points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("end_header\n")
        for p, c, n in zip(pcd.points, colors, pcd.normals):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} {n[0]} {n[1]} {n[2]}\n")
    print(f"[COLMAP] Wrote {ply}")


def prepare_colmap(rgb_frames_dir: str, raw_pcd_path: str, intrinsics_path: str,
                   extrinsics_path: str, target_w: int, target_h: int,
                   output_dir: str):
    """Prepare COLMAP-format output with raw PCD (frames already resized to target)."""
    if o3d is None:
        print("[COLMAP] Skipped (open3d not available)")
        return
    
    print(f"\n{'='*70}")
    print(f"[COLMAP] Preparing COLMAP-format output...")
    print(f"{'='*70}")
    
    sp0 = os.path.join(output_dir, "sparse", "0")
    os.makedirs(sp0, exist_ok=True)
    
    # Load trajectory & intrinsics
    traj = load_trajectory_from_log(extrinsics_path)
    K0 = np.load(intrinsics_path)
    
    # Adjust intrinsics for target resolution
    # K0 is calibrated for original sensor resolution, so we scale it to target
    K = K0.copy()
    
    # Get original resolution from intrinsics metadata or assume it
    # For now, we'll scale based on typical HOI4D resolution
    # If K0 was calibrated for full resolution, scale down for our target
    orig_w, orig_h = 1920, 1080  # Typical HOI4D original resolution
    
    sx, sy = orig_w / target_w, orig_h / target_h
    K[0, :] /= sx  # Scale fx and cx
    K[1, :] /= sy  # Scale fy and cy
    K[2, :] = [0, 0, 1]
    
    write_cameras_txt(sp0, K, target_w, target_h)
    write_images_txt(sp0, traj)
    
    # Load and save raw point cloud
    if os.path.exists(raw_pcd_path):
        pcd = o3d.io.read_point_cloud(raw_pcd_path)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=0.5)
        write_points3d_ply(sp0, pcd)
        print(f"[COLMAP] Loaded raw PCD: {len(pcd.points)} points")
    
    # Copy pre-resized RGB images to COLMAP output folder
    img_out = os.path.join(output_dir, "images")
    os.makedirs(img_out, exist_ok=True)
    print(f"[COLMAP] Copying {len(traj)} pre-resized images to {target_w}×{target_h}")
    for i in range(len(traj)):
        src = os.path.join(rgb_frames_dir, f"{i:05d}.png")
        if os.path.exists(src):
            im = Image.open(src)
            im.save(os.path.join(img_out, f"camera_rgb_{i:05d}.jpg"))
    
    print(f"[COLMAP] Saved {len(traj)} images to {img_out}")


# ============================================================================
# PART 5: NORMAL MAP GENERATION FROM DEPTH
# ============================================================================

def bilateral_filter_depth(depth: np.ndarray, sigma_spatial: float = 5.0, sigma_range: float = 0.1) -> np.ndarray:
    """Apply bilateral filter to smooth depth while preserving edges."""
    # Use OpenCV's bilateral filter
    # sigma_spatial: spatial extent of the kernel (larger = more smoothing)
    # sigma_range: range (intensity) sigma (larger = more smoothing across depth jumps)
    depth_filtered = cv2.bilateralFilter(depth.astype(np.float32), 5, sigma_range, sigma_spatial)
    return depth_filtered


def compute_depth_edge_mask(depth: np.ndarray, rgb: Optional[np.ndarray] = None, threshold: float = 0.05) -> np.ndarray:
    """
    Compute edge mask to identify depth discontinuities.
    High values = edges (unreliable normals). Low values = flat regions (reliable).
    """
    # Compute depth gradients using Sobel
    gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize to [0, 1]
    if gradient_mag.max() > 0:
        gradient_mag = gradient_mag / gradient_mag.max()
    
    # If RGB provided, also use RGB edges
    if rgb is not None:
        rgb_gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY) if rgb.shape[-1] == 3 else rgb
        gx_rgb = cv2.Sobel(rgb_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy_rgb = cv2.Sobel(rgb_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag_rgb = np.sqrt(gx_rgb**2 + gy_rgb**2)
        if gradient_mag_rgb.max() > 0:
            gradient_mag_rgb = gradient_mag_rgb / gradient_mag_rgb.max()
        gradient_mag = np.maximum(gradient_mag, gradient_mag_rgb * 0.5)  # Blend with RGB edges
    
    # Confidence: 1 - edge_mask, so edges have low confidence
    confidence = np.exp(-gradient_mag / threshold)
    return confidence


def backproject_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Back-project depth map to 3D camera coordinates.
    
    Args:
        depth: (H, W) depth map in meters
        fx, fy, cx, cy: intrinsic camera parameters
    
    Returns:
        (H, W, 3) array of 3D points in camera space [X, Y, Z]
    """
    H, W = depth.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    
    points_3d = np.stack([X, Y, Z], axis=-1)  # (H, W, 3)
    return points_3d


def compute_normals_from_depth(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                               rgb: Optional[np.ndarray] = None, smooth: bool = True,
                               use_edge_mask: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel normals using the A1 method: back-project + cross-product.
    
    Args:
        depth: (H, W) metric depth map in meters
        fx, fy, cx, cy: intrinsic camera parameters
        rgb: (H, W, 3) optional RGB image for edge detection
        smooth: whether to apply bilateral filtering to depth before computing normals
        use_edge_mask: whether to compute confidence based on depth/RGB edges
    
    Returns:
        normals: (H, W, 3) unit normal vectors in camera space, facing camera
        confidence: (H, W) confidence map (1 = reliable, 0 = unreliable at edges)
    """
    # Step 1: Smooth depth
    if smooth:
        depth_smooth = bilateral_filter_depth(depth, sigma_spatial=5.0, sigma_range=0.1)
    else:
        depth_smooth = depth.copy()
    
    # Step 2: Compute confidence/edge mask
    if use_edge_mask:
        confidence = compute_depth_edge_mask(depth_smooth, rgb=rgb, threshold=0.05)
    else:
        confidence = np.ones_like(depth_smooth)
    
    # Step 3: Back-project depth to 3D points
    points_3d = backproject_depth(depth_smooth, fx, fy, cx, cy)  # (H, W, 3)
    
    # Step 4: Compute tangent vectors using central differences
    # Pad to handle boundaries (replicate padding)
    points_padded = np.pad(points_3d, ((1, 1), (1, 1), (0, 0)), mode='edge')
    
    # t_u = X(u+1, v) - X(u-1, v) (horizontal tangent)
    t_u = points_padded[1:-1, 2:, :] - points_padded[1:-1, :-2, :]  # (H, W, 3)
    
    # t_v = X(u, v+1) - X(u, v-1) (vertical tangent)
    t_v = points_padded[2:, 1:-1, :] - points_padded[:-2, 1:-1, :]  # (H, W, 3)
    
    # Step 5: Cross product to get normal
    normals = np.cross(t_u, t_v, axisa=-1, axisb=-1, axisc=-1)  # (H, W, 3)
    
    # Step 6: Normalize
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norm_mag
    
    # Step 7: Orient normals to face camera (camera looks along -Z in camera space)
    # If n · [-Z] > 0 (i.e., n[..., 2] < 0, pointing away from camera), flip it
    camera_view = np.array([0.0, 0.0, -1.0])
    dot_product = (normals * camera_view).sum(axis=-1)  # (H, W)
    flip_mask = dot_product > 0
    normals[flip_mask] = -normals[flip_mask]
    
    return normals, confidence


def visualize_normals(normals: np.ndarray) -> np.ndarray:
    """
    Convert normal map to RGB for visualization.
    Normal components [-1, 1] → RGB [0, 255]
    """
    # Map [-1, 1] to [0, 1] to [0, 255]
    rgb_vis = ((normals + 1.0) * 127.5).astype(np.uint8)
    return rgb_vis


def save_normals(normals: np.ndarray, confidence: np.ndarray, output_dir: str, frame_idx: int):
    """Save normal map as .npy and visualization as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save binary normal map (float32)
    normal_path = os.path.join(output_dir, f"camera_normal_{frame_idx:05d}.npy")
    np.save(normal_path, normals.astype(np.float32))
    
    # Save confidence map
    conf_path = os.path.join(output_dir, f"camera_normal_confidence_{frame_idx:05d}.npy")
    np.save(conf_path, confidence.astype(np.float32))
    
    return normal_path, conf_path


def process_normals_from_depth(depth_folder: str, intrinsics_path: str, rgb_folder: Optional[str] = None,
                               output_normal_dir: str = None, output_vis_dir: str = None):
    """
    Process all depth frames to compute normal maps.
    
    Args:
        depth_folder: folder containing metric depth .png files
        intrinsics_path: path to intrinsics .npy file
        rgb_folder: optional path to RGB frames for edge detection
        output_normal_dir: folder to save normal maps (.npy files)
        output_vis_dir: folder to save normal visualizations (.png files)
    """
    if output_normal_dir is None:
        output_normal_dir = os.path.join(os.path.dirname(depth_folder), "normals")
    if output_vis_dir is None:
        output_vis_dir = os.path.join(os.path.dirname(depth_folder), "normals_vis")
    
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)
    
    # Load intrinsics
    K = np.load(intrinsics_path)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"[NORMALS] Loaded intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # Process depth frames
    depth_files = sorted([f for f in os.listdir(depth_folder) if f.endswith('.png')])
    print(f"[NORMALS] Processing {len(depth_files)} depth frames...")
    
    for idx, depth_file in enumerate(depth_files):
        depth_path = os.path.join(depth_folder, depth_file)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # Convert mm to m
        
        # Load corresponding RGB if available
        rgb = None
        if rgb_folder is not None:
            rgb_file = depth_file.replace('.png', '.png').replace('camera_depth_', '')
            rgb_path = os.path.join(rgb_folder, rgb_file)
            if os.path.exists(rgb_path):
                rgb = cv2.imread(rgb_path)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Compute normals
        normals, confidence = compute_normals_from_depth(depth, fx, fy, cx, cy, rgb=rgb, smooth=True, use_edge_mask=True)
        
        # Save normal maps
        normal_path, conf_path = save_normals(normals, confidence, output_normal_dir, idx)
        
        # Save visualization
        normal_vis = visualize_normals(normals)
        vis_path = os.path.join(output_vis_dir, f"camera_normal_{idx:05d}.png")
        cv2.imwrite(vis_path, cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))
        
        if (idx + 1) % 10 == 0:
            print(f"[NORMALS] Processed {idx + 1}/{len(depth_files)} frames")
    
    print(f"[NORMALS] Saved normal maps to {output_normal_dir}")
    print(f"[NORMALS] Saved normal visualizations to {output_vis_dir}")
    
    return output_normal_dir, output_vis_dir


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def find_dynamic_mask_folder(annotations_path: str) -> str:
    """Find the folder containing dynamic masks by searching for 'mask' in the name."""
    seg2d_path = os.path.join(annotations_path, "2Dseg")
    if not os.path.exists(seg2d_path):
        raise FileNotFoundError(f"2Dseg folder not found in {annotations_path}")

    # Search for a folder with 'mask' in its name
    for folder_name in os.listdir(seg2d_path):
        if "mask" in folder_name.lower():
            return os.path.join(seg2d_path, folder_name)

    raise FileNotFoundError(f"No folder with 'mask' in its name found in {seg2d_path}")


def resolve_hoi4d_paths(hoi4d_root: str, sequence_id: str) -> dict:
    """
    Given HOI4D root and sequence ID, resolve all required paths.
    
    Args:
        hoi4d_root: Path to parent folder containing HOI4D_release, HOI4D_depth_video, etc.
        sequence_id: Sequence identifier like "ZY20210800001/H1/C8/N11/S321/s03/T2"
    
    Returns:
        Dictionary with resolved paths (all required files must exist)
    """
    paths = {}
    
    # Remove 'annotations/' prefix if present in sequence_id
    if sequence_id.startswith('annotations/'):
        sequence_id = sequence_id[len('annotations/'):]
    
    # Extract camera_id from sequence_id (e.g., "ZY20210800001" from full path)
    camera_id = sequence_id.split('/')[0]
    
    # RGB video - from HOI4D_release
    rgb_video = os.path.join(
        hoi4d_root, "HOI4D_release", sequence_id, "align_rgb", "image.mp4"
    )
    
    # Depth video - from HOI4D_depth_video
    depth_video = os.path.join(
        hoi4d_root, "HOI4D_depth_video", sequence_id, "align_depth", "depth_video.avi"
    )
    
    # Intrinsics - from camera_params
    intrinsics = os.path.join(hoi4d_root, "camera_params", camera_id, "intrin.npy")
    
    # Extrinsics (output.log) - from HOI4D_annotations/3Dseg
    extrinsics = os.path.join(
        hoi4d_root, "HOI4D_annotations", sequence_id, "3Dseg", "output.log"
    )
    
    # Raw PCD - from HOI4D_annotations/3Dseg
    raw_pcd = os.path.join(
        hoi4d_root, "HOI4D_annotations", sequence_id, "3Dseg", "raw_pc.pcd"
    )
    
    # Dynamic masks - dynamically find the folder with 'mask' in its name
    annotations_path = os.path.join(hoi4d_root, "HOI4D_annotations", sequence_id)
    masks_folder = find_dynamic_mask_folder(annotations_path)
    
    paths['rgb_video'] = rgb_video
    paths['depth_video'] = depth_video
    paths['intrinsics'] = intrinsics
    paths['extrinsics'] = extrinsics
    paths['raw_pcd'] = raw_pcd
    paths['masks'] = masks_folder
    paths['camera_id'] = camera_id
    
    return paths


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--hoi4d_root", required=True,
        help="Root folder containing HOI4D_release, HOI4D_depth_video, camera_params, etc."
    )
    parser.add_argument(
        "--sequence_id", required=True,
        help="Sequence ID like 'ZY20210800001/H1/C8/N11/S321/s03/T2'"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output folder where all results will be saved"
    )
    
    # Optional: relative depths
    parser.add_argument(
        "--relative_depth_folder", default=None,
        help="Folder containing relative depth .npy files (per frame)"
    )
    
    # Optional: COLMAP target resolution
    parser.add_argument(
        "--colmap_width", type=int, default=None,
        help="Target width for COLMAP images (if None, use original)"
    )
    parser.add_argument(
        "--colmap_height", type=int, default=None,
        help="Target height for COLMAP images (if None, use original)"
    )
    
    # Processing parameters
    parser.add_argument(
        "--fps", type=int, default=15,
        help="FPS for video frame extraction"
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="Epochs for depth scale/shift optimization"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=10.0,
        help="Learning rate for scale/shift optimizer"
    )
    parser.add_argument(
        "--skip_depth_optimization", action="store_true",
        help="Skip depth scale/shift optimization"
    )
    parser.add_argument(
        "--skip_colmap", action="store_true",
        help="Skip COLMAP preparation"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "="*70)
    print("HOI4D DATA ADAPTATION PIPELINE")
    print("="*70)
    print(f"HOI4D Root: {args.hoi4d_root}")
    print(f"Sequence: {args.sequence_id}")
    print(f"Output: {args.output}")
    print("="*70 + "\n")
    
    # Resolve all paths
    hoi4d_paths = resolve_hoi4d_paths(args.hoi4d_root, args.sequence_id)
    
    # Verify all required files exist
    required_files = ['rgb_video', 'depth_video', 'intrinsics', 'extrinsics', 'raw_pcd', 'masks']
    for key in required_files:
        path = hoi4d_paths[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required {key}: {path}")
    
    print(f"✓ Found RGB video: {hoi4d_paths['rgb_video']}")
    print(f"✓ Found depth video: {hoi4d_paths['depth_video']}")
    print(f"✓ Found intrinsics: {hoi4d_paths['intrinsics']}")
    print(f"✓ Found extrinsics: {hoi4d_paths['extrinsics']}")
    print(f"✓ Found raw PCD: {hoi4d_paths['raw_pcd']}")
    print(f"✓ Found masks folder: {hoi4d_paths['masks']}")
    print()
    
    # ========================================================================
    # STEP 0: Determine target dimensions
    # ========================================================================
    target_w = args.colmap_width
    target_h = args.colmap_height
    
    print(f"\n{'='*70}")
    print(f"STEP 0: DETERMINE TARGET DIMENSIONS")
    print(f"{'='*70}\n")
    
    if target_w is None or target_h is None:
        # Decode first frame to get original resolution
        rgb_frames_dir_temp = os.path.join(args.output, "rgb_frames_temp")
        decode_video(hoi4d_paths['rgb_video'], rgb_frames_dir_temp, args.fps)
        sample = sorted(os.listdir(rgb_frames_dir_temp))[0]
        orig_w, orig_h = Image.open(os.path.join(rgb_frames_dir_temp, sample)).size
        if target_w is None:
            target_w = orig_w
        if target_h is None:
            target_h = orig_h
        shutil.rmtree(rgb_frames_dir_temp, ignore_errors=True)
    
    print(f"Target resolution: {target_w}×{target_h}")
    
    # ========================================================================
    # STEP 1: Decode videos to frames
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 1: DECODE VIDEOS")
    print(f"{'='*70}\n")
    
    rgb_frames_dir = os.path.join(args.output, "rgb_frames")
    depth_frames_dir = os.path.join(args.output, "depth_frames_raw")
    
    decode_video(hoi4d_paths['rgb_video'], rgb_frames_dir, args.fps)
    decode_video(hoi4d_paths['depth_video'], depth_frames_dir, args.fps)
    
    # ========================================================================
    # STEP 2: Resize RGB and depth frames to target dimensions
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 2: RESIZE FRAMES TO TARGET DIMENSIONS")
    print(f"{'='*70}\n")
    
    rgb_frames_resized = os.path.join(args.output, "rgb_frames_resized")
    depth_frames_resized = os.path.join(args.output, "depth_frames_resized")
    
    resize_images_to_target(rgb_frames_dir, rgb_frames_resized, target_w, target_h)
    print(f"[RESIZE] RGB frames → {target_w}×{target_h}")
    
    # Resize depth frames to target size (not matching RGB first, but target directly)
    os.makedirs(depth_frames_resized, exist_ok=True)
    for fn in sorted(os.listdir(depth_frames_dir)):
        if fn.endswith('.png'):
            im = Image.open(os.path.join(depth_frames_dir, fn))
            im = im.resize((target_w, target_h), Image.NEAREST)
            im.save(os.path.join(depth_frames_resized, fn))
    print(f"[RESIZE] Depth frames → {target_w}×{target_h}")
    
    # ========================================================================
    # STEP 3: Process relative depths and optimize metric depths
    # ========================================================================
    if args.relative_depth_folder:
        print(f"\n{'='*70}")
        print(f"STEP 3: PROCESS RELATIVE DEPTHS & OPTIMIZE METRIC DEPTHS")
        print(f"{'='*70}\n")
        
        rel_frames_resized = os.path.join(args.output, "relative_frames_resized")
        resize_relative_depths(args.relative_depth_folder, rel_frames_resized, target_w, target_h)
        
        if not args.skip_depth_optimization:
            rels = load_relative_depths(rel_frames_resized)
            spars = load_sparse_depths(depth_frames_resized)
            
            n = min(len(rels), len(spars))
            X_list, Y_list = [], []
            for i in range(n):
                r, s = rels[i], spars[i]
                if r.shape != s.shape:
                    s_h, s_w = s.shape[:2]
                    r = cv2.resize(r, (s_w, s_h), interpolation=cv2.INTER_NEAREST)
                rf, sf = r.flatten(), s.flatten()
                mask = (sf > 0)
                X_list.append(rf[mask])
                Y_list.append(sf[mask])
            
            X_all = np.concatenate(X_list)
            Y_all = np.concatenate(Y_list)
            print(f"[DEPTH OPT] Collected {len(X_all)} valid points for optimization")
            
            # Optimize
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[DEPTH OPT] Using device: {device}")
            a, b = optimize_scale_shift(X_all, Y_all, args.epochs, args.learning_rate, device)
            print(f"[DEPTH OPT] Learned parameters: a={a:.4f}, b={b:.4f}")
            
            # Save metric depths
            metric_depth_dir = os.path.join(args.output, "depth")
            apply_and_save_metric_depths(rels[:n], a, b, metric_depth_dir)
    
    # ========================================================================
    # STEP 4: Compute normal maps from depth
    # ========================================================================
    if args.relative_depth_folder:
        print(f"\n{'='*70}")
        print(f"STEP 4: COMPUTE NORMAL MAPS FROM METRIC DEPTH")
        print(f"{'='*70}\n")
        
        metric_depth_dir = os.path.join(args.output, "depth")
        normals_output = os.path.join(args.output, "normals")
        normals_vis_output = os.path.join(args.output, "normals_vis")
        
        if os.path.exists(metric_depth_dir):
            process_normals_from_depth(
                metric_depth_dir,
                hoi4d_paths['intrinsics'],
                rgb_folder=rgb_frames_resized,
                output_normal_dir=normals_output,
                output_vis_dir=normals_vis_output
            )
    
    # ========================================================================
    # STEP 5: Process masks from annotations
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 5: PROCESS SEGMENTATION MASKS")
    print(f"{'='*70}\n")
    
    masks_output = os.path.join(args.output, "masks")
    # Create a temporary reference with target dimensions for mask processing
    temp_ref = os.path.join(args.output, "temp_ref.png")
    Image.new('RGB', (target_w, target_h)).save(temp_ref)
    process_masks(hoi4d_paths['masks'], temp_ref, masks_output)
    os.remove(temp_ref)
    
    # ========================================================================
    # STEP 6: COLMAP preparation
    # ========================================================================
    if not args.skip_colmap:
        prepare_colmap(
            rgb_frames_resized,
            hoi4d_paths['raw_pcd'],
            hoi4d_paths['intrinsics'],
            hoi4d_paths['extrinsics'],
            target_w,
            target_h,
            args.output
        )
    
    # ========================================================================
    # STEP 7: Restructure output folders
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 7: RESTRUCTURE OUTPUT FOLDERS")
    print(f"{'='*70}\n")

    # Rename masks folder to dynamic_masks (handle existing directories)
    dynamic_masks_dir = os.path.join(args.output, "dynamic_masks")
    masks_output = os.path.join(args.output, "masks")
    if os.path.exists(masks_output):
        if os.path.exists(dynamic_masks_dir):
            shutil.rmtree(dynamic_masks_dir, ignore_errors=True)
            print(f"[RESTRUCTURE] Removed existing 'dynamic_masks' folder")
        os.rename(masks_output, dynamic_masks_dir)
        print(f"[RESTRUCTURE] Renamed 'masks' to 'dynamic_masks'")

    # Remove relative_frames_resized folder if it exists
    rel_frames_resized = os.path.join(args.output, "relative_frames_resized")
    if os.path.exists(rel_frames_resized):
        shutil.rmtree(rel_frames_resized, ignore_errors=True)
        print(f"[RESTRUCTURE] Removed 'relative_frames_resized' folder")

    # Combine sparse and images into a colmap folder (handle existing directories)
    colmap_dir = os.path.join(args.output, "colmap")
    sparse_dir = os.path.join(args.output, "sparse")
    images_dir = os.path.join(args.output, "images")
    if os.path.exists(sparse_dir) and os.path.exists(images_dir):
        if os.path.exists(colmap_dir):
            shutil.rmtree(colmap_dir, ignore_errors=True)
            print(f"[RESTRUCTURE] Removed existing 'colmap' folder")
        os.makedirs(colmap_dir, exist_ok=True)
        shutil.move(sparse_dir, os.path.join(colmap_dir, "sparse"))
        shutil.move(images_dir, os.path.join(colmap_dir, "images"))
        print(f"[RESTRUCTURE] Moved 'sparse' and 'images' into 'colmap' folder")

    # Move raw depth.avi to sparse_unprocessed_gt_depth folder (handle existing directories)
    sparse_unprocessed_dir = os.path.join(args.output, "sparse_unprocessed_gt_depth")
    if os.path.exists(sparse_unprocessed_dir):
        shutil.rmtree(sparse_unprocessed_dir, ignore_errors=True)
        print(f"[RESTRUCTURE] Removed existing 'sparse_unprocessed_gt_depth' folder")
    os.makedirs(sparse_unprocessed_dir, exist_ok=True)
    raw_depth_video = hoi4d_paths['depth_video']
    if os.path.exists(raw_depth_video):
        shutil.copy(raw_depth_video, sparse_unprocessed_dir)
        print(f"[RESTRUCTURE] Copied raw depth video to 'sparse_unprocessed_gt_depth'")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"CLEANUP")
    print(f"{'='*70}\n")
    
    # Remove temporary frame folders to save space
    print(f"[CLEANUP] Removing temporary frame folders...")
    shutil.rmtree(depth_frames_dir, ignore_errors=True)
    shutil.rmtree(rgb_frames_dir, ignore_errors=True)
    shutil.rmtree(rgb_frames_resized, ignore_errors=True)
    shutil.rmtree(depth_frames_resized, ignore_errors=True)
    
    print(f"\n{'='*70}")
    print(f"✓ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
