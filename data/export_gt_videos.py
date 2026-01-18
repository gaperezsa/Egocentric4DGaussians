#!/usr/bin/env python3
"""
Export GT videos (RGB, Depth, Normals) from a HOI4D video folder.

Usage:
    python export_gt_videos.py --video_folder /path/to/Video1
    
Example:
    python export_gt_videos.py --video_folder /home/gperezsantamaria/gperezsantamaria2/Egocentric4DGaussians/data/automatic_data_extraction_testing/with_monst3r/Video1
"""

import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import matplotlib as mpl
from PIL import Image
from tqdm import tqdm
import sys


def colorize_depth(depth_array, vmin=0.0, vmax=2.5):
    """
    Colorize depth map using jet colormap (same as metrics.py).
    
    Args:
        depth_array: numpy array (H, W) in meters
        vmin: minimum depth value for colormap
        vmax: maximum depth value for colormap
    
    Returns:
        RGB numpy array (H, W, 3) uint8
    """
    cmap = mpl.cm.get_cmap('jet')
    d = depth_array.copy()
    nonpos = d <= 0.0
    d = np.clip(d, vmin, vmax)
    denom = max(vmax - vmin, 1e-6)
    norm = (d - vmin) / denom
    vis = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    vis[nonpos] = 0  # Set invalid depths to black
    return vis


def load_depth_png(depth_path):
    """Load depth from PNG (in millimeters) and convert to meters."""
    depth_img = Image.open(depth_path)
    depth_array = np.array(depth_img).astype(np.float32) / 1000.0  # mm to meters
    return depth_array


def export_rgb_video(rgb_folder, output_path, fps=15):
    """Export RGB images to video."""
    rgb_folder = Path(rgb_folder)
    
    # Get all RGB images (sorted)
    rgb_files = sorted(list(rgb_folder.glob("camera_rgb_*.jpg")))
    if not rgb_files:
        print(f"⚠️  No RGB images found in {rgb_folder}")
        return False
    
    print(f"Found {len(rgb_files)} RGB images")
    
    # Read all images
    frames = []
    for img_path in tqdm(rgb_files, desc="Loading RGB"):
        img = imageio.imread(img_path)
        frames.append(img)
    
    # Write video
    print(f"Writing RGB video to {output_path}")
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    print(f"✓ RGB video saved: {output_path}")
    return True


def export_depth_video(depth_folder, output_path, fps=15, vmin=0.0, vmax=2.5):
    """Export depth maps to colorized video."""
    depth_folder = Path(depth_folder)
    
    # Get all depth images (sorted)
    depth_files = sorted(list(depth_folder.glob("camera_depth_*.png")))
    if not depth_files:
        print(f"⚠️  No depth images found in {depth_folder}")
        return False
    
    print(f"Found {len(depth_files)} depth images")
    
    # Read and colorize all depth maps
    frames = []
    for depth_path in tqdm(depth_files, desc="Colorizing depth"):
        depth_array = load_depth_png(depth_path)
        depth_vis = colorize_depth(depth_array, vmin=vmin, vmax=vmax)
        frames.append(depth_vis)
    
    # Write video
    print(f"Writing depth video to {output_path}")
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    print(f"✓ Depth video saved: {output_path}")
    return True


def export_depth_images(depth_folder, output_folder, vmin=0.0, vmax=2.5):
    """Export colorized depth maps as individual PNG images."""
    depth_folder = Path(depth_folder)
    output_folder = Path(output_folder)
    
    # Create output directory
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all depth images (sorted)
    depth_files = sorted(list(depth_folder.glob("camera_depth_*.png")))
    if not depth_files:
        print(f"⚠️  No depth images found in {depth_folder}")
        return False
    
    print(f"Found {len(depth_files)} depth images")
    
    # Process and save each depth map
    for depth_path in tqdm(depth_files, desc="Saving colorized depth images"):
        # Load depth
        depth_array = load_depth_png(depth_path)
        
        # Colorize depth
        depth_vis = colorize_depth(depth_array, vmin=vmin, vmax=vmax)
        
        # Generate output filename (keep same naming pattern)
        output_name = depth_path.name.replace("camera_depth_", "depth_vis_")
        output_path = output_folder / output_name
        
        # Save colorized depth image
        imageio.imwrite(output_path, depth_vis)
    
    print(f"✓ Colorized depth images saved to: {output_folder}")
    return True


def export_normals_video(normals_folder, output_path, fps=15):
    """Export normal maps to video (already colorized)."""
    normals_folder = Path(normals_folder)
    
    # Get all normal images (sorted) - note: filename is camera_normals_* (plural)
    normal_files = sorted(list(normals_folder.glob("camera_normals_*.png")))
    if not normal_files:
        print(f"⚠️  No normal images found in {normals_folder}")
        return False
    
    print(f"Found {len(normal_files)} normal images")
    
    # Read all images
    frames = []
    for img_path in tqdm(normal_files, desc="Loading normals"):
        img = imageio.imread(img_path)
        frames.append(img)
    
    # Write video
    print(f"Writing normals video to {output_path}")
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    print(f"✓ Normals video saved: {output_path}")
    return True


def export_rgb_mask_overlay_video(rgb_folder, mask_folder, output_path, fps=15, alpha=0.5):
    """
    Export RGB video with dynamic mask overlay (alpha blending).
    
    Args:
        rgb_folder: Path to RGB images folder
        mask_folder: Path to dynamic mask .npy files
        output_path: Output video path
        fps: Frames per second
        alpha: Blend factor (0.5 = 50% mask, 50% RGB)
    """
    rgb_folder = Path(rgb_folder)
    mask_folder = Path(mask_folder)
    
    # Get all RGB images (sorted)
    rgb_files = sorted(list(rgb_folder.glob("camera_rgb_*.jpg")))
    if not rgb_files:
        print(f"⚠️  No RGB images found in {rgb_folder}")
        return False
    
    # Get all mask files (sorted)
    mask_files = sorted(list(mask_folder.glob("camera_dynamics_*.npy")))
    if not mask_files:
        print(f"⚠️  No dynamic mask files found in {mask_folder}")
        return False
    
    print(f"Found {len(rgb_files)} RGB images and {len(mask_files)} mask files")
    
    # Ensure we have matching pairs
    if len(rgb_files) != len(mask_files):
        print(f"⚠️  Warning: RGB count ({len(rgb_files)}) != mask count ({len(mask_files)})")
        # Use minimum count
        count = min(len(rgb_files), len(mask_files))
        rgb_files = rgb_files[:count]
        mask_files = mask_files[:count]
    
    # Create blended frames
    frames = []
    for rgb_path, mask_path in tqdm(zip(rgb_files, mask_files), total=len(rgb_files), desc="Blending RGB+Mask"):
        # Load RGB
        rgb = imageio.imread(rgb_path)  # (H, W, 3) uint8
        
        # Load binary mask (0 or 1)
        mask_binary = np.load(mask_path)  # (H, W) with values 0 or 1
        
        # Convert mask to white (255) where dynamic, black (0) where static
        mask_vis = (mask_binary * 255).astype(np.uint8)  # (H, W)
        
        # Expand mask to 3 channels (grayscale -> RGB)
        mask_rgb = np.stack([mask_vis, mask_vis, mask_vis], axis=-1)  # (H, W, 3)
        
        # Alpha blend: result = alpha * mask + (1 - alpha) * rgb
        blended = (alpha * mask_rgb + (1 - alpha) * rgb).astype(np.uint8)
        
        frames.append(blended)
    
    # Write video
    print(f"Writing RGB+Mask overlay video to {output_path}")
    imageio.mimwrite(output_path, frames, fps=fps, codec='libx264', quality=8)
    print(f"✓ RGB+Mask overlay video saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export GT videos (RGB, Depth, Normals) from HOI4D video folder")
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Path to video folder (e.g., .../Video1)")
    parser.add_argument("--fps", type=int, default=15,
                        help="Frames per second for output videos (default: 15)")
    parser.add_argument("--depth_vmin", type=float, default=0.0,
                        help="Minimum depth value for colormap in meters (default: 0.0)")
    parser.add_argument("--depth_vmax", type=float, default=2.5,
                        help="Maximum depth value for colormap in meters (default: 2.5)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for videos (default: video_folder)")
    parser.add_argument("--mask_alpha", type=float, default=0.5,
                        help="Alpha blending factor for mask overlay (0.0-1.0, default: 0.5)")
    
    args = parser.parse_args()
    
    video_folder = Path(args.video_folder)
    if not video_folder.exists():
        print(f"❌ Error: Video folder does not exist: {video_folder}")
        sys.exit(1)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else video_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    rgb_folder = video_folder / "colmap" / "images"
    depth_folder = video_folder / "depth"
    normals_folder = video_folder / "normals_vis"
    mask_folder = video_folder / "dynamic_masks"
    depth_vis_folder = video_folder / "depth_vis"  # New folder for colorized depth images
    
    print("="*70)
    print("GT VIDEO & IMAGE EXPORT")
    print("="*70)
    print(f"Video folder: {video_folder}")
    print(f"Output directory: {output_dir}")
    print(f"FPS: {args.fps}")
    print(f"Depth range: [{args.depth_vmin}, {args.depth_vmax}] meters")
    print(f"Mask alpha: {args.mask_alpha}")
    print("="*70)
    print()
    
    # Export RGB
    print("[1/5] Exporting RGB video...")
    rgb_output = output_dir / "gt_rgb.mp4"
    rgb_success = export_rgb_video(rgb_folder, rgb_output, fps=args.fps)
    print()
    
    # Export Depth
    print("[2/5] Exporting Depth video...")
    depth_output = output_dir / "gt_depth.mp4"
    depth_success = export_depth_video(depth_folder, depth_output, 
                                       fps=args.fps, vmin=args.depth_vmin, vmax=args.depth_vmax)
    print()
    
    # Export Depth Images (NEW)
    print("[3/5] Exporting colorized depth images...")
    depth_images_success = export_depth_images(depth_folder, depth_vis_folder, 
                                               vmin=args.depth_vmin, vmax=args.depth_vmax)
    print()
    
    # Export Normals
    print("[4/5] Exporting Normals video...")
    normals_output = output_dir / "gt_normals.mp4"
    normals_success = export_normals_video(normals_folder, normals_output, fps=args.fps)
    print()
    
    # Export RGB+Mask Overlay
    print("[5/5] Exporting RGB+Mask overlay video...")
    mask_overlay_output = output_dir / "gt_rgb_mask_overlay.mp4"
    mask_success = export_rgb_mask_overlay_video(rgb_folder, mask_folder, mask_overlay_output, 
                                                   fps=args.fps, alpha=args.mask_alpha)
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    results = [
        ("RGB Video", rgb_output, rgb_success),
        ("Depth Video", depth_output, depth_success),
        ("Depth Images", depth_vis_folder, depth_images_success),
        ("Normals Video", normals_output, normals_success),
        ("RGB+Mask Video", mask_overlay_output, mask_success)
    ]
    
    for name, path, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name:13s} → {path}")
    
    total_success = sum([r[2] for r in results])
    video_count = 4  # RGB, Depth, Normals, RGB+Mask videos
    print(f"\nSuccessfully exported {total_success}/5 outputs ({video_count} videos + depth images)")
    print("="*70)
    
    return 0 if total_success == 5 else 1


if __name__ == "__main__":
    sys.exit(main())
