#!/usr/bin/env python3
"""
Visualize COLMAP Camera Centers with Point Cloud

This standalone utility loads a COLMAP sparse reconstruction and creates
a debug visualization PLY file that combines:
  1. The scene point cloud (from points3D.ply) - gray color
  2. Camera center positions (from images.txt) - bright red color

This allows visual inspection in MeshLab to verify camera trajectories
relative to the reconstructed scene.

Usage:
    python visualize_colmap_cameras.py \\
        /path/to/colmap/sparse/0 \\
        --output debug_cameras.ply
    
    # Or auto-detect sparse folder:
    python visualize_colmap_cameras.py \\
        /path/to/colmap \\
        --output debug_cameras.ply

Example 1 - ADT sequence:
    python visualize_colmap_cameras.py \\
        /home/gperezsantamaria/sda_data/Egocentric4DGaussians/data/ADT/recognition/colmap/sparse/0 \\
        --output /home/gperezsantamaria/sda_data/Egocentric4DGaussians/data/ADT/recognition/debug_cameras.ply

Example 2 - HOI4D sequence:
    python visualize_colmap_cameras.py \\
        /home/gperezsantamaria/sda_data/Egocentric4DGaussians/data/automatic_data_extraction_testing/with_monst3r/Video2/colmap/sparse \\
        --output debug_cameras.ply
"""

import argparse
import sys
from pathlib import Path
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed. Please install with:")
    print("  pip install open3d")
    sys.exit(1)


def qvec2rotmat(qvec):
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    
    Args:
        qvec: Quaternion in COLMAP format [w, x, y, z]
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R


def load_colmap_cameras_txt(images_txt_path: Path):
    """
    Load camera poses from COLMAP images.txt file.
    
    COLMAP format (images.txt):
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        POINTS2D[] (empty line)
    
    COLMAP convention (from COLMAP documentation):
        - Quaternion QW QX QY QZ: rotation from world to camera (R_wc)
        - Translation TX TY TZ: translation from world to camera (t_wc)
        - Together they form the world-to-camera transformation
        - Camera center in world coords: C = -R_wc^T * t_wc
    
    Returns:
        List of camera center positions (N, 3)
    """
    print(f"[CAMERAS] Loading camera poses from {images_txt_path.name}")
    
    camera_centers = []
    
    with open(images_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse image line (not POINTS2D line)
            parts = line.split()
            if len(parts) < 10:
                continue  # Skip POINTS2D lines
            
            # Extract pose: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            try:
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            except (ValueError, IndexError):
                continue
            
            # Convert quaternion to rotation matrix
            R_wc = qvec2rotmat(np.array([qw, qx, qy, qz]))
            t_wc = np.array([tx, ty, tz])
            
            # Compute camera center in world coordinates
            # Camera center C = -R_wc^T * t_wc
            camera_center = -R_wc.T @ t_wc
            camera_centers.append(camera_center)
    
    camera_centers = np.array(camera_centers)
    
    print(f"[CAMERAS] Loaded {len(camera_centers)} camera poses")
    return camera_centers


def load_colmap_pointcloud(points3d_ply_path: Path):
    """
    Load point cloud from COLMAP points3D.ply file.
    
    Returns:
        Open3D PointCloud object
    """
    print(f"[POINTCLOUD] Loading scene points from {points3d_ply_path.name}")
    
    if not points3d_ply_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {points3d_ply_path}")
    
    pcd = o3d.io.read_point_cloud(str(points3d_ply_path))
    
    num_points = len(pcd.points)
    print(f"[POINTCLOUD] Loaded {num_points} scene points")
    
    return pcd


def create_debug_visualization(sparse_dir: Path, output_ply: Path, 
                              mirror_x: bool = False, mirror_y: bool = False,
                              test_transform: str = None):
    """
    Create debug visualization combining COLMAP point cloud and camera centers.
    
    Args:
        sparse_dir: Path to COLMAP sparse reconstruction folder (contains points3D.ply, images.txt)
        output_ply: Output path for debug visualization PLY
        mirror_x: If True, flip camera centers across YZ plane (negate X coordinate)
        mirror_y: If True, flip camera centers across XZ plane (negate Y coordinate)
        test_transform: Experimental coordinate transformation to test
    """
    print("\n" + "="*80)
    print("COLMAP CAMERA VISUALIZATION")
    print("="*80)
    print(f"Input:  {sparse_dir}")
    print(f"Output: {output_ply}")
    if mirror_x or mirror_y:
        transforms = []
        if mirror_x:
            transforms.append("Mirror X (across YZ plane)")
        if mirror_y:
            transforms.append("Mirror Y (across XZ plane)")
        print(f"Transforms: {', '.join(transforms)}")
    if test_transform:
        print(f"Test Transform: {test_transform}")
    print("="*80)
    
    # ========== Load COLMAP data ==========
    points3d_ply = sparse_dir / "points3D.ply"
    images_txt = sparse_dir / "images.txt"
    
    # Validate files exist
    if not points3d_ply.exists():
        raise FileNotFoundError(f"Point cloud not found: {points3d_ply}")
    if not images_txt.exists():
        raise FileNotFoundError(f"Camera poses not found: {images_txt}")
    
    # Load point cloud
    scene_pcd = load_colmap_pointcloud(points3d_ply)
    scene_points = np.asarray(scene_pcd.points)
    
    # Load camera centers
    camera_centers = load_colmap_cameras_txt(images_txt)
    
    # Apply test transformation if requested
    if test_transform:
        print(f"\n[EXPERIMENTAL] Testing coordinate transformation: {test_transform}")
        print(f"[INFO] This helps determine the correct matrix to use in adapth_adt_data.py")
        print(f"[INFO] Point cloud is NOT transformed - only camera centers")
        
        original_stats = {
            'x_range': (camera_centers[:, 0].min(), camera_centers[:, 0].max()),
            'y_range': (camera_centers[:, 1].min(), camera_centers[:, 1].max()),
            'z_range': (camera_centers[:, 2].min(), camera_centers[:, 2].max()),
        }
        
        # Apply transformation
        if test_transform == 'flip_x':
            camera_centers[:, 0] = -camera_centers[:, 0]
            matrix_op = "diag([-1, 1, 1, 1])"
        elif test_transform == 'flip_y':
            camera_centers[:, 1] = -camera_centers[:, 1]
            matrix_op = "diag([1, -1, 1, 1])"
        elif test_transform == 'flip_z':
            camera_centers[:, 2] = -camera_centers[:, 2]
            matrix_op = "diag([1, 1, -1, 1])"
        elif test_transform == 'flip_xy':
            camera_centers[:, 0] = -camera_centers[:, 0]
            camera_centers[:, 1] = -camera_centers[:, 1]
            matrix_op = "diag([-1, -1, 1, 1])"
        elif test_transform == 'flip_xz':
            camera_centers[:, 0] = -camera_centers[:, 0]
            camera_centers[:, 2] = -camera_centers[:, 2]
            matrix_op = "diag([-1, 1, -1, 1])"
        elif test_transform == 'flip_yz':
            camera_centers[:, 1] = -camera_centers[:, 1]
            camera_centers[:, 2] = -camera_centers[:, 2]
            matrix_op = "diag([1, -1, -1, 1])"
        elif test_transform == 'flip_xyz':
            camera_centers[:, 0] = -camera_centers[:, 0]
            camera_centers[:, 1] = -camera_centers[:, 1]
            camera_centers[:, 2] = -camera_centers[:, 2]
            matrix_op = "diag([-1, -1, -1, 1])"
        elif test_transform == 'swap_yz':
            # Swap Y and Z axes
            temp = camera_centers[:, 1].copy()
            camera_centers[:, 1] = camera_centers[:, 2]
            camera_centers[:, 2] = temp
            matrix_op = "[X, Z, Y] = [[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]]"
        elif test_transform == 'swap_xz':
            # Swap X and Z axes
            temp = camera_centers[:, 0].copy()
            camera_centers[:, 0] = camera_centers[:, 2]
            camera_centers[:, 2] = temp
            matrix_op = "[Z, Y, X] = [[0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1]]"
        elif test_transform == 'swap_xy':
            # Swap X and Y axes
            temp = camera_centers[:, 0].copy()
            camera_centers[:, 0] = camera_centers[:, 1]
            camera_centers[:, 1] = temp
            matrix_op = "[Y, X, Z] = [[0,1,0,0], [1,0,0,0], [0,0,1,0], [0,0,0,1]]"
        elif test_transform == 'flip_xy_swap_yz':
            # First flip X and Y, then swap Y and Z
            camera_centers[:, 0] = -camera_centers[:, 0]
            camera_centers[:, 1] = -camera_centers[:, 1]
            temp = camera_centers[:, 1].copy()
            camera_centers[:, 1] = camera_centers[:, 2]
            camera_centers[:, 2] = temp
            matrix_op = "[-X, Z, -Y] = [[-1,0,0,0], [0,0,-1,0], [0,1,0,0], [0,0,0,1]]"
        elif test_transform == 'flip_xy_swap_xz':
            # First flip X and Y, then swap X and Z
            camera_centers[:, 0] = -camera_centers[:, 0]
            camera_centers[:, 1] = -camera_centers[:, 1]
            temp = camera_centers[:, 0].copy()
            camera_centers[:, 0] = camera_centers[:, 2]
            camera_centers[:, 2] = temp
            matrix_op = "[Z, -Y, -X] = [[0,0,1,0], [0,-1,0,0], [-1,0,0,0], [0,0,0,1]]"
        elif test_transform == 'swap_xy_flip_xy':
            # First swap X and Y, then flip both X and Y
            temp = camera_centers[:, 0].copy()
            camera_centers[:, 0] = camera_centers[:, 1]
            camera_centers[:, 1] = temp
            camera_centers[:, 0] = -camera_centers[:, 0]
            camera_centers[:, 1] = -camera_centers[:, 1]
            matrix_op = "[-Y, -X, Z] = [[0,-1,0,0], [-1,0,0,0], [0,0,1,0], [0,0,0,1]]"
        else:
            print(f"[ERROR] Unknown transform: {test_transform}")
            matrix_op = "identity"
        
        new_stats = {
            'x_range': (camera_centers[:, 0].min(), camera_centers[:, 0].max()),
            'y_range': (camera_centers[:, 1].min(), camera_centers[:, 1].max()),
            'z_range': (camera_centers[:, 2].min(), camera_centers[:, 2].max()),
        }
        
        print(f"\n[TRANSFORM] Original camera ranges:")
        print(f"  X: [{original_stats['x_range'][0]:.3f}, {original_stats['x_range'][1]:.3f}]")
        print(f"  Y: [{original_stats['y_range'][0]:.3f}, {original_stats['y_range'][1]:.3f}]")
        print(f"  Z: [{original_stats['z_range'][0]:.3f}, {original_stats['z_range'][1]:.3f}]")
        
        print(f"\n[TRANSFORM] After {test_transform}:")
        print(f"  X: [{new_stats['x_range'][0]:.3f}, {new_stats['x_range'][1]:.3f}]")
        print(f"  Y: [{new_stats['y_range'][0]:.3f}, {new_stats['y_range'][1]:.3f}]")
        print(f"  Z: [{new_stats['z_range'][0]:.3f}, {new_stats['z_range'][1]:.3f}]")
        
        print(f"\n[MATRIX] To apply this in adapth_adt_data.py, use:")
        print(f"  T_CORRECTION = SE3.from_matrix(np.array({matrix_op}))")
        print(f"  T_world_camera = T_world_camera_aria @ T_ARIA_TO_COLMAP @ T_CORRECTION")
        print(f"\n[INFO] If cameras look correct in MeshLab, apply this transformation!")
    
    # Apply simple mirror transformations if requested (legacy)
    elif mirror_x or mirror_y:
        print(f"\n[DEBUG] Applying mirror transformations to camera centers...")
        if mirror_x:
            print(f"  Before mirror X: X range [{camera_centers[:, 0].min():.3f}, {camera_centers[:, 0].max():.3f}]")
            camera_centers[:, 0] = -camera_centers[:, 0]
            print(f"  After mirror X:  X range [{camera_centers[:, 0].min():.3f}, {camera_centers[:, 0].max():.3f}]")
        if mirror_y:
            print(f"  Before mirror Y: Y range [{camera_centers[:, 1].min():.3f}, {camera_centers[:, 1].max():.3f}]")
            camera_centers[:, 1] = -camera_centers[:, 1]
            print(f"  After mirror Y:  Y range [{camera_centers[:, 1].min():.3f}, {camera_centers[:, 1].max():.3f}]")
    
    # ========== Print statistics ==========
    print(f"\n[DEBUG] Scene point cloud statistics:")
    print(f"  Number of points: {len(scene_points)}")
    print(f"  X range: [{scene_points[:, 0].min():.3f}, {scene_points[:, 0].max():.3f}]")
    print(f"  Y range: [{scene_points[:, 1].min():.3f}, {scene_points[:, 1].max():.3f}]")
    print(f"  Z range: [{scene_points[:, 2].min():.3f}, {scene_points[:, 2].max():.3f}]")
    print(f"  Centroid: [{scene_points[:, 0].mean():.3f}, {scene_points[:, 1].mean():.3f}, {scene_points[:, 2].mean():.3f}]")
    
    print(f"\n[DEBUG] Camera trajectory statistics:")
    print(f"  Number of cameras: {len(camera_centers)}")
    print(f"  X range: [{camera_centers[:, 0].min():.3f}, {camera_centers[:, 0].max():.3f}]")
    print(f"  Y range: [{camera_centers[:, 1].min():.3f}, {camera_centers[:, 1].max():.3f}]")
    print(f"  Z range: [{camera_centers[:, 2].min():.3f}, {camera_centers[:, 2].max():.3f}]")
    print(f"  Centroid: [{camera_centers[:, 0].mean():.3f}, {camera_centers[:, 1].mean():.3f}, {camera_centers[:, 2].mean():.3f}]")
    
    if len(camera_centers) > 0:
        print(f"  First camera: [{camera_centers[0, 0]:.3f}, {camera_centers[0, 1]:.3f}, {camera_centers[0, 2]:.3f}]")
        print(f"  Last camera:  [{camera_centers[-1, 0]:.3f}, {camera_centers[-1, 1]:.3f}, {camera_centers[-1, 2]:.3f}]")
        
        if len(camera_centers) > 1:
            motion = camera_centers[-1] - camera_centers[0]
            motion_mag = np.linalg.norm(motion)
            print(f"  Motion vector: [{motion[0]:.3f}, {motion[1]:.3f}, {motion[2]:.3f}]")
            print(f"  Motion magnitude: {motion_mag:.3f} units")
    
    # ========== Combine point clouds ==========
    print(f"\n[DEBUG] Creating combined visualization...")
    
    # Combine points
    all_points = np.vstack([scene_points, camera_centers])
    
    # Create colors: gray for scene, gradient for cameras (blue→green→red)
    scene_colors = np.ones((len(scene_points), 3)) * 0.7  # Gray
    
    # Camera colors: gradient from BLUE (start) to RED (end) via GREEN
    # Blue (0,0,1) → Cyan (0,1,1) → Green (0,1,0) → Yellow (1,1,0) → Red (1,0,0)
    num_cameras = len(camera_centers)
    camera_colors = np.zeros((num_cameras, 3))
    
    for i in range(num_cameras):
        t = i / max(num_cameras - 1, 1)  # 0.0 to 1.0
        
        if t < 0.25:
            # Blue to Cyan
            ratio = t / 0.25
            camera_colors[i] = [0, ratio, 1]
        elif t < 0.5:
            # Cyan to Green
            ratio = (t - 0.25) / 0.25
            camera_colors[i] = [0, 1, 1 - ratio]
        elif t < 0.75:
            # Green to Yellow
            ratio = (t - 0.5) / 0.25
            camera_colors[i] = [ratio, 1, 0]
        else:
            # Yellow to Red
            ratio = (t - 0.75) / 0.25
            camera_colors[i] = [1, 1 - ratio, 0]
    
    all_colors = np.vstack([scene_colors, camera_colors])
    
    # Create normals: dummy normals pointing up
    all_normals = np.zeros((len(all_points), 3))
    all_normals[:, 1] = 1.0  # All point up in Y direction
    
    # Create Open3D point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(all_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
    combined_pcd.normals = o3d.utility.Vector3dVector(all_normals)
    
    # ========== Save output ==========
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        o3d.io.write_point_cloud(str(output_ply), combined_pcd, write_ascii=True, print_progress=False)
        print(f"\n[SUCCESS] Saved debug visualization to {output_ply}")
        print(f"[INFO] Total points: {len(all_points)} ({len(scene_points)} scene + {len(camera_centers)} cameras)")
        print(f"[INFO] In MeshLab:")
        print(f"  - GRAY points = scene point cloud")
        print(f"  - Camera trajectory gradient: BLUE (start) → CYAN → GREEN → YELLOW → RED (end)")
        print("="*80)
    except Exception as e:
        print(f"\n[ERROR] Failed to save PLY: {e}")
        raise


def find_sparse_folder(input_path: Path) -> Path:
    """
    Auto-detect COLMAP sparse reconstruction folder.
    
    If input_path contains points3D.ply and images.txt, use it directly.
    Otherwise, look for sparse/0/, sparse/, or 0/ subdirectories.
    
    Args:
        input_path: User-provided path
    
    Returns:
        Path to sparse reconstruction folder containing points3D.ply and images.txt
    """
    # Check if input_path itself is the sparse folder
    if (input_path / "points3D.ply").exists() and (input_path / "images.txt").exists():
        return input_path
    
    # Common COLMAP sparse folder patterns
    candidates = [
        input_path / "sparse" / "0",
        input_path / "sparse",
        input_path / "0",
    ]
    
    for candidate in candidates:
        if candidate.exists() and (candidate / "points3D.ply").exists() and (candidate / "images.txt").exists():
            print(f"[INFO] Auto-detected sparse folder: {candidate}")
            return candidate
    
    raise FileNotFoundError(
        f"Could not find COLMAP sparse reconstruction in {input_path}\n"
        f"Expected to find points3D.ply and images.txt in:\n"
        f"  - {input_path}\n"
        f"  - {input_path}/sparse/0\n"
        f"  - {input_path}/sparse\n"
        f"  - {input_path}/0"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create debug visualization of COLMAP camera centers with point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # ADT sequence
    python visualize_colmap_cameras.py \\
        /path/to/ADT/recognition/colmap/sparse/0 \\
        --output debug_cameras.ply
    
    # HOI4D sequence (auto-detect sparse folder)
    python visualize_colmap_cameras.py \\
        /path/to/Video2/colmap \\
        --output debug_cameras.ply
    
    # Custom output location
    python visualize_colmap_cameras.py \\
        /path/to/colmap/sparse/0 \\
        --output /tmp/debug_visualization.ply

The output PLY file can be opened in MeshLab:
  - GRAY points = scene point cloud
  - BRIGHT RED points = camera center positions
        """
    )
    
    parser.add_argument(
        "colmap_path",
        type=Path,
        help="Path to COLMAP reconstruction (sparse/0 folder, or parent folder)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PLY file path (default: <colmap_path>/debug_cameras.ply)"
    )
    
    parser.add_argument(
        "--mirror-x",
        action="store_true",
        help="EXPERIMENTAL: Test mirroring camera poses across YZ plane (negate X in camera-to-world translation)"
    )
    
    parser.add_argument(
        "--mirror-y",
        action="store_true",
        help="EXPERIMENTAL: Test mirroring camera poses across XZ plane (negate Y in camera-to-world translation)"
    )
    
    parser.add_argument(
        "--test-transform",
        type=str,
        default=None,
        help="Test coordinate transformation: 'flip_x', 'flip_y', 'flip_z', 'flip_xy', 'flip_xz', 'flip_yz', 'flip_xyz', "
             "'swap_yz', 'swap_xz', 'swap_xy', 'flip_xy_swap_yz', 'flip_xy_swap_xz', 'swap_xy_flip_xy'"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.colmap_path.exists():
        print(f"ERROR: Path not found: {args.colmap_path}")
        sys.exit(1)
    
    # Find sparse reconstruction folder
    try:
        sparse_dir = find_sparse_folder(args.colmap_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Determine output path
    if args.output is None:
        output_ply = sparse_dir / "debug_cameras.ply"
    else:
        output_ply = args.output
    
    # Create visualization
    try:
        create_debug_visualization(sparse_dir, output_ply, args.mirror_x, args.mirror_y, args.test_transform)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
