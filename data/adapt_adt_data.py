#!/usr/bin/env python3
"""
ADT (Aria Digital Twin) Data Adaptation Pipeline

This script processes ADT sequences into the format expected by Egocentric4DGaussians:
  1. Extract RGB frames from VRS and undistort from fisheye to pinhole
  2. Extract sparse GT depth frames from VRS and undistort → sparse_unprocessed_gt_depth/
  3. Load MonST3R relative depth predictions (.npy files)
  4. Optimize scale/shift parameters against sparse GT
  5. Save metric depth maps → depth/
  6. Convert MPS SLAM poses to COLMAP format
  7. Extract semidense point cloud
  8. (Optional) Compute normal maps from metric depth

Input: ADT sequence folder + MonST3R depth predictions
Output: Egocentric4DGaussians-compatible data structure

Usage:
    Example 1 - Full sequence with MonST3R depths:
    python adapt_adt_data.py \\
        --adt_sequence /path/to/ADT_sequences/golden/Apartment_release_golden_skeleton_seq100_10s_sample_M1292 \\
        --output /path/to/Egocentric4DGaussians/data/ADT/seq100 \\
        --monst3r_depths /path/to/monst3r/output_ADT/video1_ADT_100frames \\
        --target_width 1408 \\
        --target_height 1408 \\
        --focal_length 610.941 \\
        --compute_normals

Example 2 - Subsample 10 seconds at 15fps (150 frames from 30fps source):
    python adapt_adt_data.py \\
        --adt_sequence /path/to/ADT_sequences/Apartment_release_clean_seq150_M1292 \\
        --output /path/to/Egocentric4DGaussians/data/ADT/clean_seq150 \\
        --monst3r_depths /path/to/monst3r/output_clean_seq150 \\
        --dynamic_masks /home/gperezsantamaria/sda_data/sam2/output_ADT/recognition/dynamic_masks \\
        --compute_normals \\
        --start_frame 0 \\
        --max_frames 150 \\
        --target_fps 15.0 \\
        --target_width 1408 \\
        --target_height 1408 \\
        --focal_length 610.941
"""

import os
import sys
import json
import gzip
import csv
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import open3d as o3d
except ImportError:
    print("Warning: open3d not installed. Point cloud processing will be skipped.")
    o3d = None

try:
    from projectaria_tools.core import data_provider, calibration, mps
    from projectaria_tools.core.stream_id import StreamId
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.image import InterpolationMethod
    from projectaria_tools.core.sophus import SE3
except ImportError:
    print("ERROR: projectaria_tools not installed. Please install with:")
    print("  pip install projectaria-tools'[all]'")
    sys.exit(1)


# ============================================================================
# COORDINATE SYSTEM TRANSFORMS
# ============================================================================

# Transform camera poses from Aria MPS SLAM coordinates to align with point cloud.
#
# IMPORTANT: The point cloud from MPS SLAM is our ground truth and stays unchanged.
# We only transform the camera poses to properly align with the point cloud.
#
# Aria MPS SLAM world frame: +X = right, +Y = down, +Z = forward
#
# After extensive testing, the Aria MPS SLAM coordinate system is already correctly
# aligned with the point cloud. No transformation is needed!
#
# Identity transformation: [X, Y, Z] → [X, Y, Z]
#
T_ARIA_TO_COLMAP = SE3.from_matrix(
    np.array([
        [ 1.0,  0.0,  0.0, 0.0],  # X = X_aria (no change)
        [ 0.0,  1.0,  0.0, 0.0],  # Y = Y_aria (no change)
        [ 0.0,  0.0,  1.0, 0.0],  # Z = Z_aria (no change)
        [ 0.0,  0.0,  0.0, 1.0],
    ])
)


# ============================================================================
# PART 1: VRS DATA EXTRACTION
# ============================================================================

def create_data_provider(vrs_file: Path):
    """Create VRS data provider from file."""
    print(f"[VRS] Opening {vrs_file.name}")
    provider = data_provider.create_vrs_data_provider(str(vrs_file))
    if not provider:
        raise RuntimeError(f"Cannot open VRS file: {vrs_file}")
    return provider


def get_camera_calibration(provider):
    """Extract camera calibration from VRS."""
    device_calib = provider.get_device_calibration()
    assert device_calib is not None, "Could not find device calibration"
    
    camera_calib = device_calib.get_camera_calib("camera-rgb")
    assert camera_calib is not None, "Could not find camera-rgb calibration"
    
    intrinsics = camera_calib.projection_params()
    width = camera_calib.get_image_size()[0].item()
    height = camera_calib.get_image_size()[1].item()
    
    return {
        'fx': intrinsics[0],
        'fy': intrinsics[0],  # Typically same for Aria
        'cx': intrinsics[1],
        'cy': intrinsics[2],
        'distortion_params': intrinsics[3:15],
        'width': width,
        'height': height,
        't_device_camera': camera_calib.get_transform_device_camera(),
        'src_calib': camera_calib,
    }


def undistort_fisheye_to_pinhole(image, src_calib, dst_calib, interpolation_method=InterpolationMethod.BILINEAR):
    """Undistort fisheye image to pinhole projection."""
    return calibration.distort_by_calibration(image, dst_calib, src_calib, interpolation_method)


# ============================================================================
# PART 2: RGB EXTRACTION & UNDISTORTION
# ============================================================================

def extract_and_undistort_rgb(provider, camera_calib, output_dir: Path, 
                              target_w: int, target_h: int, focal_length: float,
                              start_frame: int = 0, max_frames: Optional[int] = None, 
                              target_fps: Optional[float] = None):
    """
    Extract RGB frames from VRS and undistort from fisheye to pinhole.
    
    Args:
        start_frame: Index of first frame to extract (default: 0)
        max_frames: Maximum number of frames to extract (default: None = all)
        target_fps: Target FPS for output (default: None = use source FPS, no downsampling)
    
    Returns:
        List of (timestamp_ns, output_filename) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get RGB stream
    stream_id = provider.get_stream_id_from_label("camera-rgb")
    assert stream_id is not None, "Could not find camera-rgb stream"
    
    # Create pinhole calibration with device-camera transform (needed for rotation)
    dst_calib = calibration.get_linear_camera_calibration(
        target_w, target_h, focal_length, "camera-rgb", 
        camera_calib['t_device_camera']
    )
    
    total_frames = provider.get_num_data(stream_id)
    
    # Determine source FPS (estimate from first few frames)
    if target_fps is not None:
        # Get timestamps of first 10 frames to estimate FPS
        timestamps = []
        for i in range(min(10, total_frames)):
            _, metadata = provider.get_image_data_by_index(stream_id, i)
            timestamps.append(metadata.capture_timestamp_ns)
        
        # Estimate source FPS
        if len(timestamps) >= 2:
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_time_diff_ns = np.mean(time_diffs)
            source_fps = 1e9 / avg_time_diff_ns  # Convert ns to fps
            
            # Calculate sampling stride
            fps_ratio = source_fps / target_fps
            frame_stride = max(1, int(round(fps_ratio)))
            
            print(f"[RGB] Source FPS: {source_fps:.2f}, Target FPS: {target_fps:.2f}, Stride: {frame_stride}")
        else:
            frame_stride = 1
            print(f"[RGB] Warning: Could not estimate FPS, using stride=1")
    else:
        frame_stride = 1
    
    # Determine frame range
    end_frame = total_frames
    if max_frames is not None:
        # max_frames is the number of OUTPUT frames we want
        # Need to account for stride
        end_frame = min(total_frames, start_frame + max_frames * frame_stride)
    
    # Generate frame indices to extract
    frame_indices = list(range(start_frame, end_frame, frame_stride))
    
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    
    print(f"[RGB] Extracting {len(frame_indices)} frames from index {start_frame} to {frame_indices[-1] if frame_indices else start_frame}")
    print(f"[RGB] Target resolution: {target_w}×{target_h}, stride: {frame_stride}")
    
    extracted_frames = []
    
    for i in tqdm(frame_indices, desc="RGB frames"):
        # Get image data
        image_data, metadata = provider.get_image_data_by_index(stream_id, i)
        img_array = image_data.to_numpy_array()
        timestamp_ns = metadata.capture_timestamp_ns
        
        # Undistort
        undistorted = undistort_fisheye_to_pinhole(
            img_array, camera_calib['src_calib'], dst_calib
        )
        
        # Keep in raw Aria orientation (no rotation)
        # Extrinsics will naturally match this orientation
        
        # Save
        output_filename = f"camera_rgb_{timestamp_ns}.jpg"
        output_path = output_dir / output_filename
        Image.fromarray(undistorted).save(output_path, quality=95)
        
        extracted_frames.append((timestamp_ns, output_filename))
    
    print(f"[RGB] Saved {len(extracted_frames)} frames to {output_dir}")
    return extracted_frames


# ============================================================================
# PART 3: DEPTH EXTRACTION & UNDISTORTION
# ============================================================================

def extract_sparse_gt_depth(depth_provider, rgb_provider, camera_calib, 
                            rgb_frames: List[Tuple[int, str]], output_dir: Path,
                            target_w: int, target_h: int, focal_length: float):
    """
    Extract sparse GT depth frames synced with RGB and undistort.
    
    Depth is stored in depth_images.vrs with StreamId "345-1".
    Saves to sparse_unprocessed_gt_depth/ folder (in mm, uint16 PNG).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    depth_stream = StreamId("345-1")
    
    # Create pinhole calibration (same as RGB)
    dst_calib = calibration.get_linear_camera_calibration(
        target_w, target_h, focal_length, "camera-rgb"
    )
    
    print(f"[DEPTH] Extracting and undistorting {len(rgb_frames)} depth frames")
    
    extracted_count = 0
    
    for rgb_timestamp_ns, rgb_filename in tqdm(rgb_frames, desc="Depth frames"):
        # Find closest depth frame
        try:
            depth_data, depth_metadata = depth_provider.get_image_data_by_time_ns(
                depth_stream,
                rgb_timestamp_ns,
                TimeDomain.DEVICE_TIME,
                TimeQueryOptions.CLOSEST
            )
            
            depth_timestamp_ns = depth_metadata.capture_timestamp_ns
            time_diff_ns = abs(depth_timestamp_ns - rgb_timestamp_ns)
            
            # Only use if within 1 second
            if time_diff_ns > 1e9:
                continue
            
            # Get depth array (in mm, uint16)
            depth_array = depth_data.to_numpy_array()
            
            # Undistort depth using nearest neighbor (preserve integer values)
            undistorted_depth = calibration.distort_depth_by_calibration(
                depth_array, dst_calib, camera_calib['src_calib']
            )
            
            # Keep in raw Aria orientation (no rotation)
            # This matches RGB and extrinsics orientation
            
            # Save as PNG (preserves uint16)
            output_filename = f"camera_depth_{rgb_timestamp_ns}.png"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), undistorted_depth)
            
            extracted_count += 1
            
        except Exception as e:
            # Depth frame not available for this timestamp
            continue
    
    print(f"[DEPTH] Saved {extracted_count}/{len(rgb_frames)} depth frames to {output_dir}")


# ============================================================================
# PART 3: MONST3R DEPTH PROCESSING & METRIC DEPTH GENERATION
# ============================================================================

def load_sparse_depths(folder: Path) -> List[np.ndarray]:
    """Load sparse depth PNGs from folder (in mm, uint16)."""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    arrs = []
    for f in files:
        im = cv2.imread(str(folder / f), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Failed to read {f} from {folder}")
        arrs.append(im.astype(np.float32))  # in mm
    return arrs


def load_sparse_depths_as_dict(folder: Path) -> dict:
    """
    Load sparse depth PNGs and return as dict keyed by timestamp.
    
    Returns:
        dict: {timestamp_ns_str: depth_array_in_mm}
    
    Example:
        camera_depth_74878625231612.png -> {"74878625231612": array(...)}
    """
    depth_dict = {}
    for f in os.listdir(folder):
        if f.endswith('.png') and f.startswith('camera_depth_'):
            # Extract timestamp: camera_depth_74878625231612.png -> 74878625231612
            timestamp_str = f.replace('camera_depth_', '').replace('.png', '')
            
            im = cv2.imread(str(folder / f), cv2.IMREAD_UNCHANGED)
            if im is None:
                print(f"[WARNING] Failed to read {f}")
                continue
            
            depth_dict[timestamp_str] = im.astype(np.float32)  # in mm
    
    return depth_dict


def resize_relative_depths(input_folder: Path, output_dir: Path, target_w: int, target_h: int):
    """Resize relative depth .npy arrays to the target dimensions."""
    os.makedirs(output_dir, exist_ok=True)
    for fn in sorted(os.listdir(input_folder)):
        if fn.endswith('.npy'):
            arr = np.load(input_folder / fn)
            resized_arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            np.save(output_dir / fn, resized_arr)


def load_relative_depths(folder: Path) -> List[np.ndarray]:
    """Load relative depth .npy files from MonST3R output."""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy') and f.startswith('frame')])
    print(f"[MONST3R] Loading {len(files)} relative depth .npy files from {folder}")
    return [np.load(folder / f) for f in files]


def load_dynamic_masks(masks_folder: Path, num_frames: int) -> List[np.ndarray]:
    """
    Load dynamic masks (.npy files) and match them to RGB frame order.
    
    Args:
        masks_folder: Path to folder containing camera_dynamics_XXXXX.npy files
        num_frames: Expected number of frames to load
        
    Returns:
        List of boolean numpy arrays (True = dynamic object, False = background)
    """
    # Look for dynamic_masks subfolder if masks_folder is the output root
    if (masks_folder / "dynamic_masks").exists():
        masks_folder = masks_folder / "dynamic_masks"
    
    print(f"[MASKS] Loading dynamic masks from {masks_folder}")
    
    # Find all .npy mask files
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.npy')])
    
    if len(mask_files) == 0:
        raise RuntimeError(f"No .npy mask files found in {masks_folder}")
    
    print(f"[MASKS] Found {len(mask_files)} mask files")
    
    # Load masks in sorted order
    masks = []
    for fn in mask_files:
        mask = np.load(masks_folder / fn)
        # Ensure boolean type
        if mask.dtype != bool:
            mask = mask.astype(bool)
        masks.append(mask)
    
    # Verify count matches expected
    if len(masks) != num_frames:
        print(f"[MASKS] WARNING: Found {len(masks)} masks but expected {num_frames} frames")
        print(f"[MASKS] Will use first {min(len(masks), num_frames)} masks")
        masks = masks[:num_frames]
    
    print(f"[MASKS] Loaded {len(masks)} dynamic masks, shape: {masks[0].shape}")
    return masks


def optimize_scale_shift(X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, device) -> Tuple[float, float]:
    """
    Optimize scale (a) and shift (b) parameters: depth_metric = a * depth_relative + b
    
    Args:
        X: Relative depth values (N,)
        Y: Sparse GT depth values in mm (N,)
        epochs: Number of optimization epochs
        lr: Learning rate
        device: PyTorch device
        
    Returns:
        (a, b): Optimized scale and shift parameters
    """
    print(f"[DEPTH OPT] Optimizing scale/shift over {len(X)} valid points...")
    # Better initialization: use least squares estimate
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    initial_a = Y_mean / (X_mean + 1e-8)  # Rough scale estimate
    initial_b = 0.0
    
    print(f"[DEPTH OPT] Initial estimate: a={initial_a:.2f}, b={initial_b:.2f}")
    print(f"[DEPTH OPT] Input range: X=[{X.min():.4f}, {X.max():.4f}], Y=[{Y.min():.1f}, {Y.max():.1f}]")
    
    X_t = torch.from_numpy(X).float().to(device)
    Y_t = torch.from_numpy(Y).float().to(device)
    
    a = nn.Parameter(torch.tensor([initial_a], device=device))
    b = nn.Parameter(torch.tensor([initial_b], device=device))
    opt = optim.Adam([a, b], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=200, factor=0.5)
    
    best_loss = float('inf')
    for ep in range(epochs):
        opt.zero_grad()
        pred = a * X_t + b
        loss = torch.mean((pred - Y_t) ** 2)
        loss.backward()
        opt.step()
        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        if ep % 1000 == 0:
            rmse = torch.sqrt(loss).item()
            print(f"  ep {ep:5d} | lr={opt.param_groups[0]['lr']:.6f} | loss={loss.item():.2f} | RMSE={rmse:.2f} mm | a={a.item():.2f}, b={b.item():.2f}")
    
    final_a = a.item()
    final_b = b.item()
    final_rmse = np.sqrt(best_loss)
    print(f"[DEPTH OPT] Converged: a={final_a:.2f}, b={final_b:.2f}, RMSE={final_rmse:.2f} mm")
    return final_a, final_b


def colorize_depth(depth_array: np.ndarray, vmin: float = 0.0, vmax: float = 2.5) -> np.ndarray:
    """
    Colorize depth map using jet colormap.
    
    Args:
        depth_array: numpy array (H, W) in mm
        vmin: minimum depth value for colormap (in meters)
        vmax: maximum depth value for colormap (in meters)
    
    Returns:
        RGB numpy array (H, W, 3) uint8
    """
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap('jet')
    
    # Convert mm to meters for visualization
    d = depth_array.astype(np.float32) / 1000.0
    nonpos = d <= 0.0
    d = np.clip(d, vmin, vmax)
    denom = max(vmax - vmin, 1e-6)
    norm = (d - vmin) / denom
    vis = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    vis[nonpos] = 0  # Set invalid depths to black
    return vis


def apply_and_save_metric_depths(relative_list: List[np.ndarray], a: float, b: float, 
                                  out_folder: Path, rgb_frames: List[Tuple[int, str]],
                                  save_visualization: bool = True):
    """
    Apply learned scale/shift transform and save metric depth maps.
    
    Args:
        relative_list: List of relative depth arrays
        a: Scale parameter
        b: Shift parameter (in mm)
        out_folder: Output directory for metric depth PNGs
        rgb_frames: List of (timestamp_ns, filename) to match naming
        save_visualization: Whether to save colorized depth visualizations
    """
    os.makedirs(out_folder, exist_ok=True)
    
    # Create visualization folder if needed
    if save_visualization:
        vis_folder = out_folder.parent / "depth_vis"
        os.makedirs(vis_folder, exist_ok=True)
    
    print(f"[DEPTH SAVE] Saving {len(relative_list)} metric depth maps...")
    
    for i, rel in enumerate(relative_list):
        # Apply transform: metric = a * relative + b
        metric = a * rel + b
        metric = np.clip(metric, 0, None)  # Remove negative depths
        
        # Use timestamp-based naming to match RGB frames
        if i < len(rgb_frames):
            timestamp_ns, _ = rgb_frames[i]
            out_path = out_folder / f"camera_depth_{timestamp_ns}.png"
            vis_path = vis_folder / f"depth_vis_{timestamp_ns}.png" if save_visualization else None
        else:
            out_path = out_folder / f"camera_depth_{i:05d}.png"
            vis_path = vis_folder / f"depth_vis_{i:05d}.png" if save_visualization else None
        
        # Save depth as uint16 PNG (in mm)
        cv2.imwrite(str(out_path), metric.round().astype(np.uint16))
        
        # Save visualization (rotated 90° CW for natural viewing)
        if save_visualization and vis_path:
            depth_vis = colorize_depth(metric, vmin=0.0, vmax=2.5)
            # Rotate 90° CW (k=3) for visualization
            depth_vis_rotated = np.rot90(depth_vis, k=3)
            cv2.imwrite(str(vis_path), cv2.cvtColor(depth_vis_rotated, cv2.COLOR_RGB2BGR))
    
    print(f"[DEPTH SAVE] Saved metric depths to {out_folder}")
    if save_visualization:
        print(f"[DEPTH VIS] Saved depth visualizations to {vis_folder}")


def save_dynamic_masks_with_timestamps(masks: List[np.ndarray], rgb_frames: List[Tuple[int, str]], 
                                       output_dir: Path):
    """
    Save dynamic masks with timestamp-based naming to match RGB frames.
    
    Args:
        masks: List of boolean masks (True = dynamic, False = background)
        rgb_frames: List of (timestamp_ns, filename) tuples
        output_dir: Output directory for masks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[MASKS SAVE] Saving {len(masks)} dynamic masks with timestamp naming...")
    
    for i, mask in enumerate(masks):
        if i < len(rgb_frames):
            timestamp_ns, _ = rgb_frames[i]
            out_path = output_dir / f"camera_dynamics_{timestamp_ns}.npy"
        else:
            out_path = output_dir / f"camera_dynamics_{i:05d}.npy"
        
        # Save as boolean .npy
        np.save(out_path, mask)
    
    print(f"[MASKS SAVE] Saved dynamic masks to {output_dir}")


# ============================================================================
# PART 4: POSE CONVERSION (MPS → COLMAP)
# ============================================================================



# ============================================================================
# PART 4: POSE CONVERSION (MPS → COLMAP)
# ============================================================================

def load_mps_poses(trajectory_csv: Path) -> List[Dict]:
    """
    Load camera poses from MPS closed_loop_trajectory.csv.
    
    Returns list of pose dicts with timestamp_ns and transform_world_device.
    """
    print(f"[POSES] Loading MPS trajectory from {trajectory_csv.name}")
    
    closed_loop_traj = mps.read_closed_loop_trajectory(str(trajectory_csv))
    
    poses = []
    for entry in closed_loop_traj:
        timestamp_ns = int(entry.tracking_timestamp.total_seconds() * 1e9)
        poses.append({
            'timestamp_ns': timestamp_ns,
            'transform_world_device': entry.transform_world_device,
        })
    
    print(f"[POSES] Loaded {len(poses)} poses")
    return poses


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [w, x, y, z] (COLMAP format)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def generate_colmap_files(poses: List[Dict], camera_calib: Dict, 
                          rgb_frames: List[Tuple[int, str]], output_dir: Path,
                          target_w: int, target_h: int, focal_length: float):
    """
    Generate COLMAP cameras.txt and images.txt files.
    
    NOTE: Camera intrinsics account for 90° rotation applied to images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[COLMAP DEBUG] Coordinate System Transformations:")
    print(f"  1. MPS outputs: T_world_device (world = SLAM coordinate frame)")
    print(f"  2. Device→Camera: T_device_camera (from calibration)")
    print(f"  3. Need: T_world_camera for COLMAP")
    
    # Get device→camera transform
    T_device_camera = camera_calib['t_device_camera']
    print(f"\n[COLMAP DEBUG] T_device_camera from calibration:")
    print(f"  {T_device_camera.to_matrix()}")
    
    # Create rotated camera calibration (images are rotated 90° clockwise)
    # After rotation, width and height are swapped
    pinhole = calibration.get_linear_camera_calibration(
        target_w, target_h, focal_length, "camera-rgb",
        camera_calib['t_device_camera']
    )
    
    # Rotate calibration to match rotated images
    pinhole_rotated = calibration.rotate_camera_calib_cw90deg(pinhole)
    
    # ========== cameras.txt ==========
    cameras_path = output_dir / "cameras.txt"
    with open(cameras_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# PINHOLE: fx fy cx cy\n")
        
        # Images are in raw Aria orientation (no rotation)
        cx = target_w / 2.0
        cy = target_h / 2.0
        
        f.write(f"1 PINHOLE {target_w} {target_h} {focal_length} {focal_length} {cx} {cy}\n")
    
    print(f"[COLMAP] Wrote {cameras_path}")
    
    # ========== images.txt ==========
    # Match RGB frames to poses by timestamp
    pose_by_timestamp = {p['timestamp_ns']: p for p in poses}
    
    images_path = output_dir / "images.txt"
    with open(images_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as x, y, POINT3D_ID\n")
        
        for i, (timestamp_ns, filename) in enumerate(rgb_frames):
            # Find closest pose
            if not pose_by_timestamp:
                # No poses available, use identity
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                tx, ty, tz = 0.0, 0.0, 0.0
            else:
                closest_ts = min(pose_by_timestamp.keys(), key=lambda t: abs(t - timestamp_ns))
                time_diff = abs(closest_ts - timestamp_ns)
                
                if time_diff > 1e8:  # 100ms tolerance
                    # Too far, use identity
                    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                    tx, ty, tz = 0.0, 0.0, 0.0
                else:
                    pose = pose_by_timestamp[closest_ts]
                    T_world_device = pose['transform_world_device']
                    
                    # CRITICAL: Compute camera pose with coordinate frame transformation
                    # Step 1: Device → Camera (apply camera offset from device center)
                    T_world_camera_aria = T_world_device @ T_device_camera
                    
                    # Step 2: Transform camera pose to align with point cloud
                    # Transformation: [-Y, -X, Z] to match point cloud coordinate system
                    T_world_camera = T_world_camera_aria @ T_ARIA_TO_COLMAP
                    
                    # Debug first frame
                    if i == 0:
                        print(f"\n[COLMAP DEBUG] Frame {i} transformation chain:")
                        print(f"  T_world_device (from MPS SLAM, Aria frame):")
                        print(f"    {T_world_device.to_matrix()}")
                        print(f"  T_world_camera_aria (after device→camera offset, still Aria frame):")
                        print(f"    {T_world_camera_aria.to_matrix()}")
                        print(f"  T_ARIA_TO_COLMAP (identity - no transformation needed):")
                        print(f"    {T_ARIA_TO_COLMAP.to_matrix()}")
                        print(f"  T_world_camera (final camera pose):")
                        print(f"    {T_world_camera.to_matrix()}")
                    
                    # Extract rotation and translation for COLMAP format
                    # COLMAP images.txt format stores:
                    #   - Quaternion: rotation from world to camera (R_wc)
                    #   - Translation: projection center (camera position in world coordinates)
                    #
                    # We have T_world_camera (world-to-camera transform)
                    # Camera center in world coords: C = -R_wc^T * t_wc
                    # where R_wc and t_wc come from T_world_camera
                    
                    transform_matrix = T_world_camera.to_matrix()
                    R_wc = transform_matrix[:3, :3]  # Rotation: world → camera
                    t_wc = transform_matrix[:3, 3]   # Translation: world → camera
                    
                    # Compute camera center in world coordinates
                    camera_center = -R_wc.T @ t_wc
                    
                    if i == 0:
                        print(f"  T_world_camera matrix:")
                        print(f"    Rotation (world→camera):\n{R_wc}")
                        print(f"    Translation (world→camera): {t_wc}")
                        print(f"  Camera center (world coords): {camera_center}")
                        
                        # DEBUG: Check rotation matrix validity
                        det_R_wc = np.linalg.det(R_wc)
                        R_wc_T_R_wc = R_wc.T @ R_wc
                        orthogonal_error = np.linalg.norm(R_wc_T_R_wc - np.eye(3))
                        
                        print(f"\n[ROTATION DEBUG] R_wc validation:")
                        print(f"    Determinant: {det_R_wc:.6f} (should be ~1.0 for proper rotation)")
                        print(f"    R_wc^T @ R_wc - I (should be near 0):")
                        print(f"      {R_wc_T_R_wc - np.eye(3)}")
                        print(f"    Orthogonality error: {orthogonal_error:.6f}")
                        
                        # Try inverse rotation
                        R_cw = R_wc.T
                        qvec_wc = rotmat2qvec(R_wc)
                        qvec_cw = rotmat2qvec(R_cw)
                        
                        print(f"\n[ROTATION DEBUG] Quaternion comparison:")
                        print(f"    R_wc → quat (original): {qvec_wc} (w,x,y,z)")
                        print(f"    R_cw → quat (inverse):  {qvec_cw} (w,x,y,z)")
                        print(f"\n[ROTATION DEBUG] Using INVERSE (R_cw) - camera→world rotation")
                    
                    # DEBUG: Use INVERSE rotation (camera→world instead of world→camera)
                    # This might be the issue: COLMAP may expect camera-to-world, not world-to-camera
                    R_cw = R_wc.T
                    qvec = rotmat2qvec(R_cw)
                    qw, qx, qy, qz = qvec
                    tx, ty, tz = camera_center
            
            # Write image entry
            f.write(f"{i} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.6f} {ty:.6f} {tz:.6f} 1 {filename}\n")
            f.write("\n")  # Empty points2D line
    
    print(f"[COLMAP] Wrote {images_path} with {len(rgb_frames)} images")


# ============================================================================
# PART 7: POINT CLOUD EXTRACTION
# ============================================================================

def extract_semidense_pointcloud(semidense_gz: Path, output_ply: Path):
    """
    Extract semidense point cloud from MPS and save as PLY.
    
    IMPORTANT: Must filter points using confidence thresholds as per documentation.
    """
    if o3d is None:
        print("[POINTCLOUD] Skipping (open3d not installed)")
        return
    
    if not semidense_gz.exists():
        print(f"[POINTCLOUD] Warning: {semidense_gz} not found, skipping")
        return
    
    print(f"[POINTCLOUD] Loading {semidense_gz.name}")
    
    # Read MPS point cloud
    points_data = mps.read_global_point_cloud(str(semidense_gz))
    print(f"[POINTCLOUD] Loaded {len(points_data)} raw points")
    
    # CRITICAL: Filter by confidence (removes poorly estimated 3D points)
    # Recommended thresholds from documentation
    from projectaria_tools.core.mps.utils import filter_points_from_confidence
    inverse_distance_std_threshold = 0.001
    distance_std_threshold = 0.15
    
    filtered_points = filter_points_from_confidence(
        points_data, 
        inverse_distance_std_threshold, 
        distance_std_threshold
    )
    
    print(f"[POINTCLOUD] Filtered to {len(filtered_points)} high-confidence points")
    
    # Extract positions in world frame (Aria coordinate system)
    positions_aria = np.array([pt.position_world for pt in filtered_points])
    
    print(f"\n[POINTCLOUD DEBUG] Point cloud statistics (Aria frame):")
    print(f"  Number of points: {len(positions_aria)}")
    print(f"  X range: [{positions_aria[:, 0].min():.3f}, {positions_aria[:, 0].max():.3f}]")
    print(f"  Y range: [{positions_aria[:, 1].min():.3f}, {positions_aria[:, 1].max():.3f}]")
    print(f"  Z range: [{positions_aria[:, 2].min():.3f}, {positions_aria[:, 2].max():.3f}]")
    print(f"  Centroid: [{positions_aria[:, 0].mean():.3f}, {positions_aria[:, 1].mean():.3f}, {positions_aria[:, 2].mean():.3f}]")
    
    # DEBUG: Keep point cloud in original Aria coordinates (no transformation)
    # The point cloud is our ground truth - we transform camera poses to match it,
    # not the other way around!
    print(f"\n[POINTCLOUD DEBUG] Keeping point cloud in ORIGINAL Aria MPS SLAM coordinates")
    print(f"[POINTCLOUD DEBUG] Camera poses will be transformed to match this point cloud")
    positions = positions_aria  # No transformation!
    
    print(f"\n[POINTCLOUD DEBUG] Point cloud statistics (COLMAP frame):")
    print(f"  X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"  Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"  Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
    print(f"  Centroid: [{positions[:, 0].mean():.3f}, {positions[:, 1].mean():.3f}, {positions[:, 2].mean():.3f}]")
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    
    # Add random colors (COLMAP format expects colors)
    colors = np.random.rand(len(positions), 3) * 255  # 0-255 range
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D expects 0-1
    
    # Add normals (required by Egocentric4DGaussians' fetchPly function)
    # Estimate normals from point cloud geometry
    try:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print(f"[POINTCLOUD] Estimated normals for {len(positions)} points")
    except Exception as e:
        print(f"[POINTCLOUD] Warning: Normal estimation failed ({e}), continuing without normals")
    
    # Save (use ASCII format for stability)
    os.makedirs(output_ply.parent, exist_ok=True)
    try:
        o3d.io.write_point_cloud(str(output_ply), pcd, write_ascii=True, print_progress=False)
        print(f"[POINTCLOUD] Saved {len(positions)} points to {output_ply}")
    except Exception as e:
        print(f"[POINTCLOUD] Error saving PLY: {e}")
        raise


# ============================================================================
# PART 7: NORMAL MAP COMPUTATION (OPTIONAL)
# ============================================================================

def compute_normals_from_depth_batch(depth_dir: Path, output_normal_dir: Path, 
                                     output_vis_dir: Path, fx: float, fy: float, 
                                     cx: float, cy: float):
    """
    Compute normal maps from depth frames.
    
    Uses back-projection and cross-product method.
    NOTE: Depth maps are rotated 90° clockwise, so intrinsics must match.
    """
    os.makedirs(output_normal_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)
    
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    
    if not depth_files:
        print("[NORMALS] No depth files found, skipping")
        return
    
    print(f"[NORMALS] Computing normals for {len(depth_files)} frames")
    
    for depth_fn in tqdm(depth_files, desc="Normal maps"):
        depth_path = depth_dir / depth_fn
        
        # Load depth (in mm, convert to meters)
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        
        # Compute normals (depth is already rotated, so use rotated intrinsics)
        normals = compute_normals_from_depth_map(depth_m, fx, fy, cx, cy)
        
        # Save .npy (singular: camera_normal_*)
        timestamp = depth_fn.replace('camera_depth_', '').replace('.png', '')
        normal_npy_path = output_normal_dir / f"camera_normal_{timestamp}.npy"
        np.save(normal_npy_path, normals)
        
        # Save visualization (rotated 90° CW for natural viewing)
        normal_vis = visualize_normals(normals)
        # Rotate 90° CW (k=3) for visualization
        normal_vis_rotated = np.rot90(normal_vis, k=3)
        normal_vis_path = output_vis_dir / f"camera_normal_{timestamp}.png"
        cv2.imwrite(str(normal_vis_path), normal_vis_rotated)
    
    print(f"[NORMALS] Saved to {output_normal_dir}")


def compute_normals_from_depth_map(depth: np.ndarray, fx: float, fy: float, 
                                   cx: float, cy: float) -> np.ndarray:
    """
    Compute surface normals from depth map using back-projection + cross-product.
    
    Returns:
        normals: (H, W, 3) unit normal vectors
    """
    h, w = depth.shape
    
    # Back-project to 3D
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    Z = depth
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points_3d = np.stack([X, Y, Z], axis=-1)
    
    # Compute tangent vectors via central differences
    points_padded = np.pad(points_3d, ((1, 1), (1, 1), (0, 0)), mode='edge')
    t_u = points_padded[1:-1, 2:, :] - points_padded[1:-1, :-2, :]  # Horizontal
    t_v = points_padded[2:, 1:-1, :] - points_padded[:-2, 1:-1, :]  # Vertical
    
    # Cross product
    normals = np.cross(t_u, t_v, axisa=-1, axisb=-1, axisc=-1)
    
    # Normalize
    norm_mag = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm_mag + 1e-8)
    
    # Flip to face camera (camera looks down -Z)
    camera_direction = np.array([0.0, 0.0, -1.0])
    dot_product = np.sum(normals * camera_direction[np.newaxis, np.newaxis, :], axis=-1)
    flip_mask = dot_product > 0
    normals[flip_mask] *= -1
    
    return normals


def visualize_normals(normals: np.ndarray) -> np.ndarray:
    """Convert normal map to RGB visualization."""
    vis = ((normals + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def process_adt_sequence(adt_path: Path, output_path: Path, 
                         monst3r_depths: Path,
                         dynamic_masks: Path,
                         target_w: int = 1408, target_h: int = 1408, 
                         focal_length: float = 610.941, 
                         compute_normals: bool = False,
                         start_frame: int = 0,
                         max_frames: Optional[int] = None,
                         target_fps: Optional[float] = None):
    """
    Main pipeline to convert ADT sequence to Egocentric4DGaussians format.
    
    Args:
        adt_path: Path to ADT sequence folder
        output_path: Output directory
        monst3r_depths: Path to MonST3R output folder with relative depth .npy files
        dynamic_masks: Path to dynamic masks folder with binary .npy files
        start_frame: Index of first RGB frame to extract (default: 0)
        max_frames: Maximum number of output frames to extract (default: None = all)
        target_fps: Target FPS for output (default: None = use source FPS)
    """
    print("="*80)
    print("ADT DATA ADAPTATION PIPELINE")
    print("="*80)
    print(f"Input:  {adt_path}")
    print(f"Output: {output_path}")
    print(f"MonST3R depths: {monst3r_depths}")
    print(f"Dynamic masks: {dynamic_masks}")
    print(f"Target resolution: {target_w}×{target_h}, focal length: {focal_length}")
    if max_frames is not None or start_frame > 0 or target_fps is not None:
        print(f"Frame sampling: start={start_frame}, max_frames={max_frames}, target_fps={target_fps}")
    print("="*80)
    
    # ========== Setup paths ==========
    vrs_rgb = adt_path / "video.vrs"
    vrs_depth = adt_path / "depth_images.vrs"
    
    mps_traj = adt_path / "mps" / "slam" / "closed_loop_trajectory.csv"
    mps_points = adt_path / "mps" / "slam" / "semidense_points.csv.gz"
    
    # Validate required files
    required_files = [vrs_rgb, vrs_depth, mps_traj]
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")
    
    # Validate MonST3R depths folder
    if not monst3r_depths.exists():
        raise FileNotFoundError(f"MonST3R depths folder not found: {monst3r_depths}")
    
    npy_files = list(monst3r_depths.glob("frame_*.npy"))
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No frame_*.npy files found in {monst3r_depths}")
    
    # Create output directories
    colmap_images_dir = output_path / "colmap" / "images"
    colmap_sparse_dir = output_path / "colmap" / "sparse" / "0"
    sparse_gt_dir = output_path / "sparse_unprocessed_gt_depth"
    depth_dir = output_path / "depth"
    normals_dir = output_path / "normals"
    normals_vis_dir = output_path / "normals_vis"
    
    # ========== Step 1: Extract & undistort RGB ==========
    print("\n" + "="*80)
    print("STEP 1: RGB EXTRACTION & UNDISTORTION")
    print("="*80)
    
    provider_rgb = create_data_provider(vrs_rgb)
    camera_calib = get_camera_calibration(provider_rgb)
    
    rgb_frames = extract_and_undistort_rgb(
        provider_rgb, camera_calib, colmap_images_dir,
        target_w, target_h, focal_length,
        start_frame, max_frames, target_fps
    )
    
    # ========== Step 2: Extract sparse GT depth ==========
    print("\n" + "="*80)
    print("STEP 2: SPARSE GT DEPTH EXTRACTION (VRS → sparse_unprocessed_gt_depth/)")
    print("="*80)
    
    provider_depth = create_data_provider(vrs_depth)
    
    extract_sparse_gt_depth(
        provider_depth, provider_rgb, camera_calib, rgb_frames,
        sparse_gt_dir, target_w, target_h, focal_length
    )
    
    # ========== Step 3: Load MonST3R relative depths ==========
    print("\n" + "="*80)
    print("STEP 3: LOAD MONST3R RELATIVE DEPTHS")
    print("="*80)
    
    relative_depths = load_relative_depths(monst3r_depths)
    
    # Rotate MonST3R depths to match raw Aria orientation (90° CCW)
    print(f"[MONST3R] Rotating relative depths 90° CCW to match Aria orientation")
    relative_depths_rotated = []
    for depth in relative_depths:
        # Rotate 90° CCW (k=1) to match raw Aria
        rotated = np.rot90(depth, k=1)
        relative_depths_rotated.append(rotated)
    relative_depths = relative_depths_rotated
    
    # Check if relative depths need resizing (after rotation)
    if len(relative_depths) > 0:
        rel_h, rel_w = relative_depths[0].shape
        # Now in Aria orientation (no swap needed)
        
        if rel_h != target_h or rel_w != target_w:
            print(f"[MONST3R] Resizing relative depths from {rel_w}×{rel_h} to {target_w}×{target_h}")
            resized_depths = []
            for depth in relative_depths:
                resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                resized_depths.append(resized)
            relative_depths = resized_depths
    
    # ========== Step 3b: Load dynamic masks ==========
    print("\n" + "="*80)
    print("STEP 3b: LOAD DYNAMIC MASKS")
    print("="*80)
    
    masks = load_dynamic_masks(dynamic_masks, len(rgb_frames))
    
    # Rotate dynamic masks 90° CCW to match raw Aria orientation
    print(f"[MASKS] Rotating dynamic masks 90° CCW to match Aria orientation")
    masks_rotated = []
    for mask in masks:
        # Rotate 90° CCW (k=1) to match raw Aria
        rotated_mask = np.rot90(mask, k=1)
        masks_rotated.append(rotated_mask)
    masks = masks_rotated
    
    # Verify mask dimensions match relative depths (after rotation)
    if len(masks) > 0 and len(relative_depths) > 0:
        mask_h, mask_w = masks[0].shape
        rel_h, rel_w = relative_depths[0].shape
        if mask_h != rel_h or mask_w != rel_w:
            print(f"[MASKS] WARNING: Mask shape {mask_w}×{mask_h} doesn't match depth shape {rel_w}×{rel_h}")
            print(f"[MASKS] Resizing masks to match depth...")
            resized_masks = []
            for mask in masks:
                resized_mask = cv2.resize(mask.astype(np.uint8), (rel_w, rel_h), 
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
                resized_masks.append(resized_mask)
            masks = resized_masks
            print(f"[MASKS] Resized {len(masks)} masks to {rel_w}×{rel_h}")
    
    # ========== Step 4: Optimize scale/shift against sparse GT (background only) ==========
    print("\n" + "="*80)
    print("STEP 4: OPTIMIZE SCALE/SHIFT (relative → metric, background only)")
    print("="*80)
    
    # Load sparse GT depths as dict keyed by timestamp
    sparse_depths_dict = load_sparse_depths_as_dict(sparse_gt_dir)
    print(f"[DEPTH OPT] Loaded {len(sparse_depths_dict)} sparse GT depth frames")
    
    # Collect valid correspondence points (where sparse GT is non-zero AND mask is background)
    # Now properly paired by timestamp!
    X_list = []  # Relative depth values
    Y_list = []  # Sparse GT depth values (mm)
    
    total_sparse_points = 0
    filtered_points = 0
    matched_frames = 0
    
    for i, (timestamp_ns, rgb_filename) in enumerate(rgb_frames):
        # Extract timestamp string from RGB filename: camera_rgb_74876425582900.jpg -> 74876425582900
        timestamp_str = str(timestamp_ns)
        
        # Check if we have sparse GT for this timestamp
        if timestamp_str not in sparse_depths_dict:
            # No sparse GT for this RGB frame, skip
            continue
        
        # Get the paired data
        rel = relative_depths[i]  # Relative depth for this RGB frame (correct pairing)
        sparse = sparse_depths_dict[timestamp_str]  # Sparse GT for this RGB frame (correct pairing)
        
        matched_frames += 1
        
        valid_mask = sparse > 0  # Non-zero sparse GT
        total_sparse_points += np.sum(valid_mask)
        
        # Filter out dynamic regions (mask=True means dynamic object)
        if i < len(masks):
            background_mask = ~masks[i]  # Invert: True = background, False = dynamic
            
            # Ensure masks are same shape as depth
            if background_mask.shape != sparse.shape:
                background_mask = cv2.resize(background_mask.astype(np.uint8), 
                                            (sparse.shape[1], sparse.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
            
            # Combine: valid sparse GT AND background
            combined_mask = valid_mask & background_mask
        else:
            # No mask available for this frame, use all valid points
            combined_mask = valid_mask
        
        num_valid = np.sum(combined_mask)
        filtered_points += num_valid
        
        if num_valid > 0:
            X_list.append(rel[combined_mask].flatten())
            Y_list.append(sparse[combined_mask].flatten())
    
    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    
    print(f"[DEPTH OPT] Total RGB frames: {len(rgb_frames)}")
    print(f"[DEPTH OPT] Matched frames with sparse GT: {matched_frames}/{len(rgb_frames)} ({100*matched_frames/len(rgb_frames):.1f}%)")
    print(f"[DEPTH OPT] Total sparse GT points: {total_sparse_points}")
    print(f"[DEPTH OPT] Background points (after filtering dynamic): {filtered_points}")
    print(f"[DEPTH OPT] Filtered out {total_sparse_points - filtered_points} dynamic points ({100*(total_sparse_points - filtered_points)/max(total_sparse_points, 1):.1f}%)")
    print(f"[DEPTH OPT] Using {len(X)} background points for optimization")
    
    # Optimize scale/shift
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a, b = optimize_scale_shift(X, Y, epochs=10000, lr=1.0, device=device)
    
    # ========== Step 5: Save metric depths ==========
    print("\n" + "="*80)
    print("STEP 5: SAVE METRIC DEPTH MAPS (depth/ + depth_vis/)")
    print("="*80)
    
    apply_and_save_metric_depths(relative_depths, a, b, depth_dir, rgb_frames, save_visualization=True)
    
    # ========== Step 5b: Save dynamic masks with timestamps ==========
    print("\n" + "="*80)
    print("STEP 5b: SAVE DYNAMIC MASKS (dynamic_masks/)")
    print("="*80)
    
    dynamic_masks_dir = output_path / "dynamic_masks"
    save_dynamic_masks_with_timestamps(masks, rgb_frames, dynamic_masks_dir)
    
    # ========== Step 6: Convert poses to COLMAP ==========
    print("\n" + "="*80)
    print("STEP 6: POSE CONVERSION (MPS → COLMAP)")
    print("="*80)
    
    poses = load_mps_poses(mps_traj)
    
    generate_colmap_files(
        poses, camera_calib, rgb_frames, colmap_sparse_dir,
        target_w, target_h, focal_length
    )
    
    # ========== Step 7: Extract point cloud ==========
    print("\n" + "="*80)
    print("STEP 7: POINT CLOUD EXTRACTION")
    print("="*80)
    
    output_ply = colmap_sparse_dir / "points3D.ply"
    extract_semidense_pointcloud(mps_points, output_ply)
    
    # ========== Step 7b: Create debug visualization ==========
    print("\n" + "="*80)
    print("STEP 7b: DEBUG VISUALIZATION (point cloud + camera centers)")
    print("="*80)
    
    print("[DEBUG VIS] Use standalone script to visualize camera centers:")
    print(f"  python visualize_colmap_cameras.py {colmap_sparse_dir} --output debug_cameras.ply")
    
    # ========== Step 8: Compute normals (optional) ==========
    if compute_normals:
        print("\n" + "="*80)
        print("STEP 8: NORMAL MAP COMPUTATION (from metric depth)")
        print("="*80)
        
        # Compute normals with raw Aria intrinsics (no rotation)
        cx = target_w / 2.0
        cy = target_h / 2.0
        
        compute_normals_from_depth_batch(
            depth_dir, normals_dir, normals_vis_dir,
            focal_length, focal_length, cx, cy
        )
    
    # ========== Step 7: Create train/test splits ==========
    # print("\n" + "="*80)
    # print("STEP 7: TRAIN/TEST SPLIT GENERATION")
    # print("="*80)
    # 
    # split_dir = output_path / "split"
    # create_splits(bbox_csv, len(rgb_frames), split_dir)
    
    # ========== Done! ==========
    print("\n" + "="*80)
    print("✅ ADT DATA ADAPTATION COMPLETE")
    print("="*80)
    print(f"Output directory: {output_path}")
    print(f"  - RGB frames: {colmap_images_dir}")
    print(f"  - Sparse GT depth: {sparse_gt_dir}")
    print(f"  - Metric depth: {depth_dir}")
    print(f"  - Dynamic masks: {output_path / 'dynamic_masks'}")
    print(f"  - COLMAP files: {colmap_sparse_dir}")
    if compute_normals:
        print(f"  - Normal maps: {normals_dir}")
    print("="*80)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert ADT sequence to Egocentric4DGaussians format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example 1 - Full sequence:
    python adapt_adt_data.py \\
        --adt_sequence /path/to/ADT_sequences/golden/Apartment_release_golden_skeleton_seq100_10s_sample_M1292 \\
        --output /path/to/Egocentric4DGaussians/data/ADT/seq100 \\
        --target_width 1408 \\
        --target_height 1408 \\
        --focal_length 610.941 \\
        --compute_normals

Example 2 - Subsample 10 seconds at 15fps (150 frames from 30fps source):
    python adapt_adt_data.py \\
        --adt_sequence /path/to/ADT_sequences/Apartment_release_clean_seq150_M1292 \\
        --output /path/to/Egocentric4DGaussians/data/ADT/clean_seq150 \\
        --start_frame 0 \\
        --max_frames 150 \\
        --target_fps 15.0 \\
        --target_width 1408 \\
        --target_height 1408 \\
        --focal_length 610.941
        """
    )
    
    parser.add_argument(
        "--adt_sequence",
        type=Path,
        required=True,
        help="Path to ADT sequence folder (contains video.vrs, depth_images.vrs, etc.)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--monst3r_depths",
        type=Path,
        required=True,
        help="Path to MonST3R output folder containing relative depth .npy files (frame_XXXXX.npy)"
    )
    
    parser.add_argument(
        "--dynamic_masks",
        type=Path,
        required=True,
        help="Path to dynamic masks folder containing binary .npy files per frame (camera_dynamics_XXXXX.npy)"
    )
    
    parser.add_argument(
        "--target_width",
        type=int,
        default=1408,
        help="Target image width after undistortion (default: 1408)"
    )
    
    parser.add_argument(
        "--target_height",
        type=int,
        default=1408,
        help="Target image height after undistortion (default: 1408)"
    )
    
    parser.add_argument(
        "--focal_length",
        type=float,
        default=610.941,
        help="Focal length for pinhole camera (default: 610.941)"
    )
    
    parser.add_argument(
        "--compute_normals",
        action="store_true",
        help="Compute normal maps from depth (optional)"
    )
    
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="Index of first RGB frame to extract (default: 0)"
    )
    
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Maximum number of output frames to extract (default: None = all frames)"
    )
    
    parser.add_argument(
        "--target_fps",
        type=float,
        default=None,
        help="Target FPS for output video (default: None = use source FPS). "
             "For example, if source is 30fps and you want 15fps, set this to 15.0"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.adt_sequence.exists():
        print(f"ERROR: ADT sequence not found: {args.adt_sequence}")
        sys.exit(1)
    
    if not args.monst3r_depths.exists():
        print(f"ERROR: MonST3R depths folder not found: {args.monst3r_depths}")
        sys.exit(1)
    
    if not args.dynamic_masks.exists():
        print(f"ERROR: Dynamic masks folder not found: {args.dynamic_masks}")
        sys.exit(1)
    
    # Run pipeline
    try:
        process_adt_sequence(
            args.adt_sequence,
            args.output,
            args.monst3r_depths,
            args.dynamic_masks,
            args.target_width,
            args.target_height,
            args.focal_length,
            args.compute_normals,
            args.start_frame,
            args.max_frames,
            args.target_fps
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
