"""
DN-Splatter Utilities for Normal and Depth Regularization

This module implements the depth and normal regularization techniques from:
"DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing"
https://arxiv.org/pdf/2403.17822

Key features:
1. Gradient-aware depth loss with logarithmic penalty (Eq. 4)
2. Geometric normal computation from Gaussian parameters (Eq. 6-7)
3. Scale regularization to encourage disc-like Gaussians (Eq. 8)
4. Normal regularization with monocular priors and TV smoothness (Eq. 10-11)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


def compute_image_gradient(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute image gradient magnitude for gradient-aware weighting.
    
    Args:
        image: RGB image tensor of shape (3, H, W) or (B, 3, H, W)
        eps: Small constant for numerical stability
        
    Returns:
        Gradient magnitude tensor of shape (H, W) or (B, H, W)
    """
    if image.ndim == 3:
        # Single image (3, H, W) -> add batch dimension
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Convert to grayscale if RGB (take mean across channels)
    if image.shape[1] == 3:
        gray = image.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    else:
        gray = image
    
    # Compute gradients using Sobel-like filters
    # Sobel kernel for x-direction
    sobel_x = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    
    # Sobel kernel for y-direction  
    sobel_y = torch.tensor([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    
    # Apply convolution with padding
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    # Compute gradient magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    
    # Remove channel dimension
    grad_mag = grad_mag.squeeze(1)  # (B, H, W)
    
    if squeeze_output:
        grad_mag = grad_mag.squeeze(0)  # (H, W)
    
    return grad_mag


def compute_gaussian_normals(quaternions: torch.Tensor, 
                             scales: torch.Tensor,
                             means3D: Optional[torch.Tensor] = None,
                             camera_center: Optional[torch.Tensor] = None,
                             flip_to_camera: bool = True) -> torch.Tensor:
    """
    Compute geometric normals from Gaussian parameters (Eq. 6-7 from DN-Splatter).
    
    The normal is defined by the rotation matrix R (from quaternion) applied to a one-hot vector
    indicating the minimum scaling axis (the axis along which the Gaussian is thinnest).
    
    Args:
        quaternions: Gaussian quaternions (N, 4) in format (w, x, y, z)
        scales: Gaussian scaling parameters (N, 3)
        means3D: Gaussian center positions (N, 3) in world space (optional, for flipping)
        camera_center: Camera center position (3,) in world space (optional, for flipping)
        flip_to_camera: If True and means3D/camera_center provided, flip normals to face camera
        
    Returns:
        Normal vectors (N, 3) in world space
    """
    N = quaternions.shape[0]
    device = quaternions.device
    
    # Normalize quaternions
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    
    # Extract quaternion components (w, x, y, z)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Build rotation matrix from quaternion (Eq. 7)
    # R = [[1-2(y²+z²),  2(xy-wz),    2(xz+wy)   ],
    #      [2(xy+wz),    1-2(x²+z²),  2(yz-wx)   ],
    #      [2(xz-wy),    2(yz+wx),    1-2(x²+y²) ]]
    
    R = torch.zeros((N, 3, 3), device=device, dtype=quaternions.dtype)
    
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y*z - w*x)
    
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    # Find the axis with minimum scaling (thinnest axis) - Eq. 6
    min_scale_idx = torch.argmin(scales, dim=1)  # (N,)
    
    # Create one-hot vectors for minimum scaling axis
    one_hot = torch.zeros((N, 3), device=device, dtype=scales.dtype)
    one_hot.scatter_(1, min_scale_idx.unsqueeze(1), 1.0)
    
    # Compute normal: n_i = R · OneHot(argmin(s_1, s_2, s_3))
    normals = torch.bmm(R, one_hot.unsqueeze(-1)).squeeze(-1)  # (N, 3)
    
    # Normalize to unit vectors
    normals = F.normalize(normals, p=2, dim=-1)
    
    # Optionally flip normals to face the camera
    if flip_to_camera and means3D is not None and camera_center is not None:
        # Compute viewing direction for each Gaussian
        view_dirs = camera_center.unsqueeze(0) - means3D  # (N, 3)
        view_dirs = F.normalize(view_dirs, p=2, dim=-1)
        
        # Compute dot product
        dot_product = (normals * view_dirs).sum(dim=-1)  # (N,)
        
        # Flip normals where dot product is negative
        flip_mask = dot_product < 0
        normals[flip_mask] = -normals[flip_mask]
    
    return normals


def scale_regularization_loss(scales: torch.Tensor, 
                              lambda_scale: float = 0.01) -> torch.Tensor:
    """
    Regularization to encourage disc-like Gaussians by minimizing the smallest scaling axis (Eq. 8).
    
    From DN-Splatter: "We minimize one of the scaling axes during training 
    to force Gaussians to become disc-like surfels."
    
    L_scale = sum_i ||argmin(s_i)||_1
    
    Args:
        scales: Gaussian scaling parameters (N, 3)
        lambda_scale: Weight for scale regularization
        
    Returns:
        Scalar loss encouraging disc-like shapes
    """
    # Find minimum scale for each Gaussian
    min_scales = torch.min(scales, dim=1).values  # (N,)
    
    # L1 penalty on minimum scales (Eq. 8)
    loss = lambda_scale * min_scales.abs().mean()
    
    return loss


def gradient_aware_depth_loss(pred_depth: torch.Tensor, 
                               gt_depth: torch.Tensor,
                               rgb_image: Optional[torch.Tensor] = None,
                               image_gradient: Optional[torch.Tensor] = None,
                               mask: Optional[torch.Tensor] = None,
                               depth_threshold: float = 0.001,
                               eps: float = 1e-8) -> torch.Tensor:
    """
    Gradient-aware depth loss from DN-Splatter (Eq. 4).
    
    Uses g_rgb = exp(-∇I) to reduce loss at image edges and enforce it more on smooth regions.
    Applies logarithmic penalty: log(1 + ||pred - gt||_1)
    
    Args:
        pred_depth: Predicted depth map (H, W) or (B, H, W)
        gt_depth: Ground truth depth map (H, W) or (B, H, W)
        rgb_image: Aligned RGB image (3, H, W) or (B, 3, H, W). Either this or image_gradient must be provided.
        image_gradient: Pre-computed image gradient (H, W) or (B, H, W). If provided, rgb_image is ignored.
        mask: Optional valid depth mask (H, W) or (B, H, W). If None, all valid depths > threshold are used.
        depth_threshold: Minimum valid depth value
        eps: Small constant for numerical stability
        
    Returns:
        Scalar loss tensor
    """
    # Ensure batch dimension
    if pred_depth.ndim == 2:
        pred_depth = pred_depth.unsqueeze(0)
        gt_depth = gt_depth.unsqueeze(0)
    
    # Get gradient magnitude (either pre-computed or compute now)
    if image_gradient is not None:
        grad_mag = image_gradient
        if grad_mag.ndim == 2:
            grad_mag = grad_mag.unsqueeze(0)
    else:
        assert rgb_image is not None, "Either rgb_image or image_gradient must be provided"
        grad_mag = compute_image_gradient(rgb_image, eps=eps)
        if grad_mag.ndim == 2:
            grad_mag = grad_mag.unsqueeze(0)
    
    # Compute gradient-aware weight: g_rgb = exp(-∇I)
    g_rgb = torch.exp(-grad_mag)


def normal_regularization_loss(pred_normals: torch.Tensor,
                               gt_normals: torch.Tensor,
                               rgb_image: Optional[torch.Tensor] = None,
                               image_gradient: Optional[torch.Tensor] = None,
                               mask: Optional[torch.Tensor] = None,
                               lambda_l1: float = 1.0,
                               lambda_tv: float = 0.01,
                               use_gradient_aware: bool = True) -> Tuple[torch.Tensor, Dict]:
    """
    Normal regularization loss from DN-Splatter (Eq. 10-11) with optional gradient-aware weighting.
    
    Combines:
    1. L1 loss with monocular normal priors (optionally gradient-aware like depth loss)
    2. Total variation (TV) smoothness prior
    
    Args:
        pred_normals: Predicted/rendered normal map (3, H, W) or (B, 3, H, W)
        gt_normals: Ground truth normal map from monocular estimator (3, H, W) or (B, 3, H, W)
        rgb_image: Aligned RGB image (3, H, W) or (B, 3, H, W). Required if use_gradient_aware=True and image_gradient is None.
        image_gradient: Pre-computed image gradient (H, W) or (B, H, W)
        mask: Optional valid region mask (H, W) or (B, H, W)
        lambda_l1: Weight for L1 normal loss
        lambda_tv: Weight for TV smoothness loss
        use_gradient_aware: If True, apply exp(-∇I) weighting to reduce loss at edges
        
    Returns:
        Tuple of (total_loss, dict with individual components)
    """
    # Ensure batch dimension
    if pred_normals.ndim == 3:
        pred_normals = pred_normals.unsqueeze(0)
        gt_normals = gt_normals.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, C, H, W = pred_normals.shape
    assert C == 3, "Normal maps must have 3 channels"
    
    # --- L1 Loss with GT normals (Eq. 10) with optional gradient-aware weighting ---
    l1_diff = torch.abs(pred_normals - gt_normals)  # (B, 3, H, W)
    
    # Apply gradient-aware weighting if requested
    if use_gradient_aware:
        # Get gradient magnitude
        if image_gradient is not None:
            grad_mag = image_gradient
            if grad_mag.ndim == 2:
                grad_mag = grad_mag.unsqueeze(0)
        else:
            assert rgb_image is not None, "Either rgb_image or image_gradient must be provided for gradient-aware weighting"
            grad_mag = compute_image_gradient(rgb_image)
            if grad_mag.ndim == 2:
                grad_mag = grad_mag.unsqueeze(0)
        
        # Compute gradient-aware weight: g_rgb = exp(-∇I)
        g_rgb = torch.exp(-grad_mag)  # (B, H, W)
        # Expand to match normal channels
        g_rgb_expanded = g_rgb.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, H, W)
        
        # Weight the L1 difference
        l1_diff = l1_diff * g_rgb_expanded
    
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # Expand mask to match channels
        mask_expanded = mask.unsqueeze(1).expand(-1, 3, -1, -1)  # (B, 3, H, W)
        l1_loss = l1_diff[mask_expanded].mean() if mask_expanded.any() else torch.tensor(0.0, device=pred_normals.device)
    else:
        l1_loss = l1_diff.mean()
    
    # --- Total Variation (TV) Smoothness Loss (Eq. 11) ---
    # Compute finite differences along x and y axes
    # ∇_i N_{i,j} ≈ N_{i+1,j} - N_{i,j}
    # ∇_j N_{i,j} ≈ N_{i,j+1} - N_{i,j}
    
    # Gradient along height (vertical)
    diff_i = torch.abs(pred_normals[:, :, 1:, :] - pred_normals[:, :, :-1, :])  # (B, 3, H-1, W)
    
    # Gradient along width (horizontal)
    diff_j = torch.abs(pred_normals[:, :, :, 1:] - pred_normals[:, :, :, :-1])  # (B, 3, H, W-1)
    
    # Sum of absolute gradients
    tv_loss = diff_i.mean() + diff_j.mean()
    
    # --- Combined Loss ---
    total_loss = lambda_l1 * l1_loss + lambda_tv * tv_loss
    
    loss_dict = {
        'normal_l1': l1_loss.item(),
        'normal_tv': tv_loss.item(),
        'normal_total': total_loss.item()
    }
    
    return total_loss, loss_dict


def render_normal_map_from_gaussians(gaussians, viewpoint_camera, pipe, bg_color, stage="fine", cam_type=None):
    """
    Render a normal map by computing normals from Gaussian geometry and using them as colors.
    
    This uses the same rendering pipeline but replaces colors with normals computed from
    the Gaussian's rotation and scale parameters (Eq. 6-7).
    
    Args:
        gaussians: GaussianModel instance
        viewpoint_camera: Camera to render from
        pipe: Pipeline parameters
        bg_color: Background color tensor
        stage: Training stage
        cam_type: Camera type
        
    Returns:
        rendered_normals: (3, H, W) tensor of rendered normals in camera space
        visibility_filter: Boolean mask of visible Gaussians
        radii: Radii of projected Gaussians
    """
    from gaussian_renderer import render
    
    # Get Gaussian parameters
    means3D = gaussians.get_xyz
    scales = gaussians.get_scaling  # Already activated
    rotations = gaussians.get_rotation  # Already activated (quaternions)
    
    # Apply deformation if needed
    if stage != "coarse" and hasattr(gaussians, '_deformation'):
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)
        means3D, scales, rotations, _, _ = gaussians._deformation(
            means3D, gaussians._scaling, gaussians._rotation, 
            gaussians._opacity, gaussians.get_features, time
        )
        scales = gaussians.scaling_activation(scales)
        rotations = gaussians.rotation_activation(rotations)
    
    # Compute normals from Gaussian geometry (Eq. 6-7)
    normals_world = compute_gaussian_normals(
        quaternions=rotations,
        scales=scales,
        means3D=means3D,
        camera_center=viewpoint_camera.camera_center.cuda(),
        flip_to_camera=True
    )
    
    # Transform normals to camera space
    # Camera rotation is the 3x3 upper-left of world_view_transform
    R_w2c = viewpoint_camera.world_view_transform[:3, :3].cuda()
    normals_cam = (R_w2c @ normals_world.T).T  # (N, 3)
    
    # Normalize and map from [-1, 1] to [0, 1] for rendering
    normals_cam = F.normalize(normals_cam, p=2, dim=-1)
    normals_rgb = (normals_cam + 1.0) / 2.0  # Map to [0, 1]
    
    # Render using normals as colors
    render_pkg = render(
        viewpoint_camera=viewpoint_camera,
        pc=gaussians,
        pipe=pipe,
        bg_color=bg_color,
        override_color=normals_rgb,
        stage=stage,
        cam_type=cam_type
    )
    
    # Map rendered image back to [-1, 1] normal range
    rendered_normals = render_pkg["render"] * 2.0 - 1.0  # (3, H, W)
    
    # Renormalize to unit length
    rendered_normals = F.normalize(rendered_normals, p=2, dim=0)
    
    return rendered_normals, render_pkg["visibility_filter"], render_pkg["radii"]

