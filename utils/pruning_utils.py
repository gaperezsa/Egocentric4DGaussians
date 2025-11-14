"""
Pruning utilities for Gaussian splatting models.
"""

import torch
import numpy as np
import os
from PIL import Image
import cv2


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
