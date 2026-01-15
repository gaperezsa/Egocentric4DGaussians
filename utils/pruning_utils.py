def zero_static_gaussian_grads(gaussians):
    """
    Zero out gradients for all static (background) gaussians.
    Call this after backward() and before optimizer.step() if static freezing is enabled.
    """
    if not (hasattr(gaussians, '_freeze_static') and gaussians._freeze_static):
        return
    static_mask = ~gaussians._dynamic_xyz
    if gaussians._xyz.grad is not None:
        gaussians._xyz.grad[static_mask] = 0.0
    if gaussians._features_dc.grad is not None:
        gaussians._features_dc.grad[static_mask] = 0.0
    if gaussians._features_rest.grad is not None:
        gaussians._features_rest.grad[static_mask] = 0.0
    if gaussians._opacity.grad is not None:
        gaussians._opacity.grad[static_mask] = 0.0
    if gaussians._scaling.grad is not None:
        gaussians._scaling.grad[static_mask] = 0.0
    if gaussians._rotation.grad is not None:
        gaussians._rotation.grad[static_mask] = 0.0
"""
Pruning utilities for Gaussian splatting models.
"""

import torch
import numpy as np
import os
from PIL import Image
import cv2

"""
Chamfer-based pruning utilities for dynamic Gaussians.

This module provides pruning strategies based on Chamfer distance at canonical time,
removing dynamic Gaussians that contribute most to the reconstruction error.
"""

import torch
import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def compute_chamfer_distances_per_gaussian(dynamic_xyz, gt_pointcloud):
    """
    Compute per-Gaussian Chamfer distances to ground truth point cloud.
    
    For each dynamic Gaussian, finds the distance to its nearest GT point.
    This identifies which Gaussians are "outliers" far from the true dynamic object.
    
    Args:
        dynamic_xyz: [M, 3] tensor of dynamic Gaussian positions (deformed or canonical)
        gt_pointcloud: [N, 3] tensor of ground truth 3D points from dynamic mask
    
    Returns:
        distances: [M] tensor of nearest-neighbor distances for each Gaussian
    """
    # Compute pairwise distances: [M, N]
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    dynamic_xyz_sq = (dynamic_xyz ** 2).sum(dim=1, keepdim=True)  # [M, 1]
    gt_pointcloud_sq = (gt_pointcloud ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    
    # [M, N] = [M, 1] + [1, N] - 2 * [M, 3] @ [3, N]
    pairwise_sq = dynamic_xyz_sq + gt_pointcloud_sq.T - 2.0 * (dynamic_xyz @ gt_pointcloud.T)
    pairwise_sq = torch.clamp(pairwise_sq, min=0.0)  # Numerical stability
    
    # For each Gaussian, find distance to nearest GT point
    min_distances, _ = pairwise_sq.min(dim=1)  # [M]
    distances = torch.sqrt(min_distances)
    
    return distances


def prune_high_chamfer_dynamic_gaussians(
    gaussians, 
    scene, 
    stage, 
    gt_pointcloud_cache,
    prune_percent=0.20,
    num_time_samples=5
):
    """
    Prune dynamic Gaussians that contribute most to Chamfer loss at canonical/near-canonical time.
    
    Strategy:
    1. Get the first N training cameras closest to t=0.
    2. For each camera, deform Gaussians to its specific time.
    3. Compute per-Gaussian distance to that camera's GT point cloud.
    4. Aggregate distances across all sampled cameras (max distance).
    5. Prune top X% of Gaussians with highest distances.
    
    Args:
        gaussians: GaussianModel with dynamic Gaussians and deformation network
        scene: Scene object containing cameras
        stage: Current training stage (should be "dynamics_depth")
        gt_pointcloud_cache: Dict mapping camera uid -> [N, 3] GT point cloud tensor
        prune_percent: Fraction of worst Gaussians to remove (0.0 to 1.0)
        num_time_samples: Number of cameras to sample near canonical time
    """
    device = gaussians.get_xyz.device
    n_gaussians = gaussians.get_xyz.shape[0]
    
    # Get dynamic Gaussian mask
    dynamic_mask = gaussians._dynamic_xyz  # [N] boolean
    n_dynamic = dynamic_mask.sum().item()
    
    if n_dynamic == 0:
        print("  No dynamic Gaussians found, skipping Chamfer-based pruning")
        return
    
    print(f"\n{'='*80}")
    print(f"[CHAMFER-BASED PRUNING] Removing top {prune_percent*100:.1f}% worst dynamic Gaussians")
    print(f"{'='*80}")
    print(f"  Total Gaussians: {n_gaussians}")
    print(f"  Dynamic Gaussians: {n_dynamic}")
    
    # Get canonical positions of dynamic Gaussians
    dynamic_indices = torch.where(dynamic_mask)[0]
    canonical_xyz = gaussians.get_xyz[dynamic_indices]  # [M, 3]
    
    # Get training cameras sorted by proximity to t=0
    train_cams = list(scene.getTrainCameras())
    train_cams_sorted = sorted(train_cams, key=lambda cam: abs(cam.time))  # Sort by distance from t=0
    
    # Use the first num_time_samples cameras that are in the cache
    reference_cameras = []
    for cam in train_cams_sorted:
        if cam.uid in gt_pointcloud_cache:
            reference_cameras.append(cam)
        if len(reference_cameras) >= num_time_samples:
            break
            
    if not reference_cameras:
        print("  WARNING: No cameras found in gt_pointcloud_cache, skipping pruning")
        return

    print(f"  Using {len(reference_cameras)} cameras near t=0 for GT reference")
    
    # Track per-Gaussian distances across all sampled cameras
    # We'll use the MAXIMUM distance (worst case) to identify consistent outliers
    max_distances_per_gaussian = torch.zeros(n_dynamic, device=device)
    
    # For each reference camera
    for cam in reference_cameras:
        t_sample = cam.time
        cam_id = cam.uid
        gt_pointcloud = gt_pointcloud_cache[cam_id]
        
        if gt_pointcloud.shape[0] == 0:
            continue
            
        print(f"  Sampling camera {cam_id} at t={t_sample:.4f}...")
        
        # Deform Gaussians to this time using the deformation network
        with torch.no_grad():
            if hasattr(gaussians, '_deformation') and gaussians._deformation is not None:
                # Get deformation at this time - must be (N, 1) shape to match render path
                time_tensor = torch.full((n_dynamic, 1), t_sample, device=device)
                
                # Get other Gaussian parameters for dynamic Gaussians
                scales = gaussians._scaling[dynamic_indices]
                rotations = gaussians._rotation[dynamic_indices]
                opacity = gaussians._opacity[dynamic_indices]
                shs = gaussians._features_dc[dynamic_indices]
                
                # Apply deformation network
                deformed_xyz, _, _, _, _ = gaussians._deformation(
                    canonical_xyz,
                    scales,
                    rotations,
                    opacity,
                    shs,
                    time_tensor
                )
            else:
                # No deformation network (shouldn't happen in dynamics_depth, but handle gracefully)
                deformed_xyz = canonical_xyz
        
        # Compute per-Gaussian distances to nearest GT point
        distances = compute_chamfer_distances_per_gaussian(deformed_xyz, gt_pointcloud)  # [M]
        
        # Update max distances (track worst case for each Gaussian)
        max_distances_per_gaussian = torch.max(max_distances_per_gaussian, distances)
        
        # Log statistics for this camera
        print(f"    Mean distance: {distances.mean().item():.4f}")
        print(f"    Max distance:  {distances.max().item():.4f}")
    
    # Identify top X% worst Gaussians (highest Chamfer distances)
    n_to_prune = int(n_dynamic * prune_percent)
    
    if n_to_prune == 0:
        print(f"\n  ✓ Prune percent too low, no Gaussians to remove")
        print(f"{'='*80}\n")
        return
    
    # Get indices of top N worst Gaussians
    _, worst_indices = torch.topk(max_distances_per_gaussian, n_to_prune, largest=True)
    
    # Convert to global Gaussian indices
    worst_global_indices = dynamic_indices[worst_indices]
    
    # Create pruning mask
    prune_mask = torch.zeros(n_gaussians, dtype=torch.bool, device=device)
    prune_mask[worst_global_indices] = True
    
    # Log statistics
    print(f"\n  Distance statistics for dynamic Gaussians:")
    print(f"    Mean:   {max_distances_per_gaussian.mean().item():.4f}")
    print(f"    Median: {max_distances_per_gaussian.median().item():.4f}")
    print(f"    Max:    {max_distances_per_gaussian.max().item():.4f}")
    print(f"    Min:    {max_distances_per_gaussian.min().item():.4f}")
    
    # Show threshold for pruning
    prune_threshold = max_distances_per_gaussian[worst_indices[-1]].item()
    print(f"\n  Pruning {n_to_prune}/{n_dynamic} dynamic Gaussians with distance > {prune_threshold:.4f}")
    print(f"  Keeping {n_dynamic - n_to_prune} dynamic Gaussians")
    
    # Optional: Save visualization of distance distribution
    debug_dir = os.path.join(scene.model_path, "pruning_debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save histogram of distances
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        distances_cpu = max_distances_per_gaussian.cpu().numpy()
        plt.hist(distances_cpu, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(prune_threshold, color='r', linestyle='--', linewidth=2, label=f'Prune threshold: {prune_threshold:.4f}')
        plt.xlabel('Distance to nearest GT point')
        plt.ylabel('Number of Gaussians')
        plt.title(f'Chamfer Distance Distribution (Pruning top {prune_percent*100:.1f}%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(debug_dir, f"chamfer_distances_prune_{prune_percent*100:.0f}pct.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved distance histogram to {debug_dir}/chamfer_distances_prune_{prune_percent*100:.0f}pct.png")
    except Exception as e:
        print(f"  WARNING: Failed to save histogram: {e}")
    
    # Prune the worst dynamic Gaussians
    gaussians.prune_points(prune_mask)
    
    # Report final counts
    n_after = gaussians.get_xyz.shape[0]
    n_dynamic_after = gaussians._dynamic_xyz.sum().item()
    print(f"\n  Final counts: {n_after} total ({n_gaussians - n_after} removed), {n_dynamic_after} dynamic")
    print(f"{'='*80}\n")


def prune_high_chamfer_dynamic_gaussians_adaptive(
    gaussians, 
    scene, 
    stage, 
    gt_pointcloud_cache,
    distance_threshold=0.05,
    num_time_samples=5
):
    """
    Adaptive variant: prune based on absolute distance threshold rather than percentage.
    
    Strategy:
    1. Get the first N training cameras closest to t=0.
    2. For each camera, deform Gaussians to its specific time.
    3. Compute per-Gaussian distance to that camera's GT point cloud.
    4. Aggregate distances across all sampled cameras (max distance).
    5. Prune ALL Gaussians beyond the distance threshold.
    
    Args:
        distance_threshold: Remove Gaussians with max distance > this value (in world units)
        (other args same as prune_high_chamfer_dynamic_gaussians)
    """
    device = gaussians.get_xyz.device
    n_gaussians = gaussians.get_xyz.shape[0]
    
    dynamic_mask = gaussians._dynamic_xyz
    n_dynamic = dynamic_mask.sum().item()
    
    if n_dynamic == 0:
        print("  No dynamic Gaussians found, skipping adaptive Chamfer-based pruning")
        return
    
    print(f"\n{'='*80}")
    print(f"[ADAPTIVE CHAMFER PRUNING] Removing Gaussians with distance > {distance_threshold:.4f}")
    print(f"{'='*80}")
    print(f"  Total Gaussians: {n_gaussians}")
    print(f"  Dynamic Gaussians: {n_dynamic}")
    
    # Get canonical positions of dynamic Gaussians
    dynamic_indices = torch.where(dynamic_mask)[0]
    canonical_xyz = gaussians.get_xyz[dynamic_indices]
    
    # Get training cameras sorted by proximity to t=0
    train_cams = list(scene.getTrainCameras())
    train_cams_sorted = sorted(train_cams, key=lambda cam: abs(cam.time))
    
    # Use the first num_time_samples cameras that are in the cache
    reference_cameras = []
    for cam in train_cams_sorted:
        if cam.uid in gt_pointcloud_cache:
            reference_cameras.append(cam)
        if len(reference_cameras) >= num_time_samples:
            break
            
    if not reference_cameras:
        print("  WARNING: No cameras found in gt_pointcloud_cache, skipping pruning")
        return
    
    max_distances_per_gaussian = torch.zeros(n_dynamic, device=device)
    
    for cam in reference_cameras:
        t_sample = cam.time
        cam_id = cam.uid
        gt_pointcloud = gt_pointcloud_cache[cam_id]
        
        if gt_pointcloud.shape[0] == 0:
            continue
            
        with torch.no_grad():
            if hasattr(gaussians, '_deformation') and gaussians._deformation is not None:
                time_tensor = torch.full((n_dynamic, 1), t_sample, device=device)
                scales = gaussians._scaling[dynamic_indices]
                rotations = gaussians._rotation[dynamic_indices]
                opacity = gaussians._opacity[dynamic_indices]
                shs = gaussians.get_features[dynamic_indices]
                
                deformed_xyz, _, _, _, _ = gaussians._deformation(
                    canonical_xyz, scales, rotations, opacity, shs, time_tensor
                )
            else:
                deformed_xyz = canonical_xyz
        
        distances = compute_chamfer_distances_per_gaussian(deformed_xyz, gt_pointcloud)
        max_distances_per_gaussian = torch.max(max_distances_per_gaussian, distances)
    
    # Prune based on absolute threshold
    outlier_indices = torch.where(max_distances_per_gaussian > distance_threshold)[0]
    n_to_prune = len(outlier_indices)
    
    if n_to_prune == 0:
        print(f"\n  ✓ No Gaussians exceed distance threshold, no pruning needed")
        print(f"{'='*80}\n")
        return
    
    # Convert to global indices
    outlier_global_indices = dynamic_indices[outlier_indices]
    
    # Create pruning mask
    prune_mask = torch.zeros(n_gaussians, dtype=torch.bool, device=device)
    prune_mask[outlier_global_indices] = True
    
    print(f"\n  Pruning {n_to_prune}/{n_dynamic} dynamic Gaussians (distance > {distance_threshold:.4f})")
    print(f"  Keeping {n_dynamic - n_to_prune} dynamic Gaussians")
    
    # Prune
    gaussians.prune_points(prune_mask)
    
    n_after = gaussians.get_xyz.shape[0]
    n_dynamic_after = gaussians._dynamic_xyz.sum().item()
    print(f"\n  Final counts: {n_after} total ({n_gaussians - n_after} removed), {n_dynamic_after} dynamic")
    print(f"{'='*80}\n")


def prune_invisible_dynamic_gaussians(gaussians, scene, pipe, background, stage, num_cameras=5):
    """
    Prune dynamic Gaussians that are not visible inside dynamic masks at canonical time (t=0).
    
    For each of the first N training cameras, this function:
    1. Renders only dynamic Gaussians
    2. Checks which Gaussians were actually splatted (radii > 0)
    3. For each splatted Gaussian, checks if its screen-space position overlaps with the dynamic mask
    4. Marks Gaussians that NEVER overlap with any dynamic mask across all cameras for removal
    
    This ensures only valid dynamic Gaussians (visible inside dynamic regions at t=0)
    are passed to the next stage (dynamics_RGB).
    
    Args:
        gaussians: GaussianModel with dynamic Gaussians spawned
        scene: Scene object containing cameras
        pipe: Pipeline parameters
        background: Background color tensor
        stage: Current training stage (should be "dynamics_depth")
        num_cameras: Number of cameras to test visibility against (default: 5)
    """
    from gaussian_renderer import render_with_dynamic_gaussians_mask
    
    device = gaussians.get_xyz.device
    n_gaussians = gaussians.get_xyz.shape[0]
    
    # Get dynamic Gaussian mask
    dynamic_mask = gaussians._dynamic_xyz  # [N] boolean
    n_dynamic = dynamic_mask.sum().item()
    
    if n_dynamic == 0:
        print("  No dynamic Gaussians found, skipping pruning")
        return
    
    print(f"  Total Gaussians: {n_gaussians}")
    print(f"  Dynamic Gaussians: {n_dynamic}")
    
    # Track which dynamic Gaussians are valid (visible inside dynamic mask)
    ever_valid = torch.zeros(n_gaussians, dtype=torch.bool, device=device)
    
    # Get first N training cameras (sorted by time for canonical time proximity)
    train_cams = list(scene.getTrainCameras())
    train_cams_sorted = sorted(train_cams, key=lambda cam: cam.time)
    validation_cameras = train_cams_sorted[:num_cameras]
    
    print(f"  Validating with {len(validation_cameras)} cameras at t={validation_cameras[0].time:.3f} to t={validation_cameras[-1].time:.3f}")
    
    # Create debug directory
    debug_dir = os.path.join(scene.model_path, "pruning_debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Render each camera with only dynamic Gaussians
    for cam_idx, viewpoint in enumerate(validation_cameras):
        # Check if camera has a dynamic mask
        if not hasattr(viewpoint, 'dynamic_mask') or viewpoint.dynamic_mask is None:
            print(f"    Camera {cam_idx}: No dynamic mask available, skipping")
            continue
        
        # Get ground truth dynamic mask for this camera
        gt_dynamic_mask = viewpoint.dynamic_mask  # [H, W] boolean tensor
        H, W = gt_dynamic_mask.shape
        
        # Move mask to GPU if needed
        if not gt_dynamic_mask.is_cuda:
            gt_dynamic_mask = gt_dynamic_mask.to(device)
        
        # Create override_opacity: 1.0 for dynamic, 0.0 for static
        override_opacity = dynamic_mask.to(gaussians._opacity.dtype).unsqueeze(1)  # [N, 1]
        
        # Render with dynamic Gaussians only
        with torch.no_grad():
            render_pkg = render_with_dynamic_gaussians_mask(
                viewpoint, gaussians, pipe, background,
                stage=stage, 
                cam_type=scene.dataset_type,
                override_opacity=override_opacity,
                training=False  # Disable gradients
            )
        
        # PROJECT ALL DYNAMIC GAUSSIAN CENTERS TO THIS CAMERA'S IMAGE PLANE
        # Simple approach: check if center projects inside the dynamic mask
        
        # Get all dynamic Gaussian 3D positions
        dynamic_indices = torch.where(dynamic_mask)[0]
        dynamic_3d_pos = gaussians.get_xyz[dynamic_indices]  # [M, 3]
        
        # Convert to homogeneous coordinates
        ones = torch.ones(dynamic_3d_pos.shape[0], 1, device=device)
        points_world_homog = torch.cat([dynamic_3d_pos, ones], dim=1)  # [M, 4]
        
        # Transform to camera space using world_view_transform
        world_view = viewpoint.world_view_transform.cuda()  # [4, 4]
        points_camera_homog = (world_view @ points_world_homog.T).T  # [M, 4]
        points_camera = points_camera_homog[:, :3]  # [M, 3] - (x, y, z) in camera space
        
        # Manual perspective projection using field of view
        # tan(FoV/2) gives us the scaling factor
        import math
        tan_fovx = math.tan(viewpoint.FoVx * 0.5)
        tan_fovy = math.tan(viewpoint.FoVy * 0.5)
        
        # Project: x_ndc = (x_cam / z_cam) / tan(fovx/2), y_ndc = (y_cam / z_cam) / tan(fovy/2)
        z_cam = points_camera[:, 2:3]  # [M, 1]
        x_ndc = (points_camera[:, 0:1] / (z_cam + 1e-10)) / tan_fovx
        y_ndc = (points_camera[:, 1:2] / (z_cam + 1e-10)) / tan_fovy
        
        # Convert from NDC [-1, 1] to pixel coordinates [0, W] x [0, H]
        pixel_x = ((x_ndc[:, 0] + 1.0) * 0.5 * W).long()
        pixel_y = ((y_ndc[:, 0] + 1.0) * 0.5 * H).long()
        
        # Clamp to image bounds
        pixel_x = torch.clamp(pixel_x, 0, W - 1)
        pixel_y = torch.clamp(pixel_y, 0, H - 1)
        
        # Check which Gaussian centers fall inside the dynamic mask
        # gt_dynamic_mask[pixel_y, pixel_x] gives True if that pixel is in the dynamic region
        inside_mask = gt_dynamic_mask[pixel_y, pixel_x]  # [M] boolean
        
        # Mark these Gaussians as valid (have been seen inside a dynamic mask)
        ever_valid[dynamic_indices[inside_mask]] = True
        
        num_inside = inside_mask.sum().item()
        num_outside = (~inside_mask).sum().item()
        
        print(f"    Camera {cam_idx} (t={viewpoint.time:.3f}): {num_inside} inside mask, {num_outside} outside")
        
        # DEBUG: Save visualization
        rendered_img = render_pkg["render"]  # [3, H, W]
        
        # Convert to numpy for visualization
        rendered_img_np = (rendered_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Get dynamic mask as image
        mask_viz = (gt_dynamic_mask.cpu().numpy() * 255).astype(np.uint8)
        mask_viz = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)
        
        # DEBUG: Create visualization showing Gaussian centers
        debug_img = rendered_img_np.copy()
        
        # Draw Gaussian centers (first 100 only to avoid clutter)
        pixel_x_cpu = pixel_x.cpu().numpy()
        pixel_y_cpu = pixel_y.cpu().numpy()
        inside_mask_cpu = inside_mask.cpu().numpy()
        
        for g_idx in range(min(100, len(dynamic_indices))):
            px, py = pixel_x_cpu[g_idx], pixel_y_cpu[g_idx]
            is_inside = inside_mask_cpu[g_idx]
            
            # Draw center point
            color = (0, 255, 0) if is_inside else (255, 0, 0)  # Green if inside mask, red otherwise
            if 0 <= px < W and 0 <= py < H:
                cv2.circle(debug_img, (int(px), int(py)), 2, color, -1)
        
        # Save debug images
        debug_prefix = f"cam{cam_idx:02d}_t{viewpoint.time:.3f}"
        
        # Save combined view
        combined = np.hstack([rendered_img_np, mask_viz, cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)])
        Image.fromarray(combined).save(os.path.join(debug_dir, f"{debug_prefix}_combined.png"))
    
    # Determine which Gaussians to prune:
    # - Dynamic Gaussians that were NEVER inside a dynamic mask across all cameras
    never_valid_dynamic = dynamic_mask & (~ever_valid)
    n_to_prune = never_valid_dynamic.sum().item()
    
    if n_to_prune == 0:
        print(f"\n  ✓ All {n_dynamic} dynamic Gaussians are valid (inside masks), no pruning needed")
        return
    
    print(f"\n  Pruning {n_to_prune}/{n_dynamic} dynamic Gaussians that were never inside dynamic masks")
    print(f"  Keeping {n_dynamic - n_to_prune} valid dynamic Gaussians")
    
    # Prune the invalid dynamic Gaussians
    gaussians.prune_points(never_valid_dynamic)
    
    # Report final counts
    n_after = gaussians.get_xyz.shape[0]
    n_dynamic_after = gaussians._dynamic_xyz.sum().item()
    print(f"  Final counts: {n_after} total ({n_gaussians - n_after} removed), {n_dynamic_after} dynamic")
