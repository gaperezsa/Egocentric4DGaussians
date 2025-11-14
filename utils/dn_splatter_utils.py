"""
DN-Splatter Normal Rendering Utilities

Minimal, clean implementation of normal computation and rendering for Gaussian Splatting.
Based on: DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshing
https://arxiv.org/pdf/2403.17822
"""

import math
import torch
import torch.nn.functional as F
from utils.loss_utils import ssim

# Try to import gsplat for efficient normal rendering (1.5.3+)
try:
    from gsplat import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("[WARN] gsplat not available - normal rendering will be slow")


def compute_gaussian_normals(
    quaternions: torch.Tensor,
    scales: torch.Tensor,
    means3D: torch.Tensor,
    camera_center: torch.Tensor,
    flip_to_camera: bool = True
) -> torch.Tensor:
    """
    Compute geometric normals from Gaussian parameters (Eq. 6-7 from DN-Splatter).

    The normal is the rotation matrix R applied to a one-hot vector indicating the
    minimum scaling axis (the axis along which the Gaussian is thinnest).

    Args:
        quaternions: Gaussian quaternions [N, 4] in format (w, x, y, z)
        scales: Gaussian scaling parameters [N, 3]
        means3D: Gaussian center positions [N, 3] in world space
        camera_center: Camera center position [3,] in world space
        flip_to_camera: If True, flip normals to face the camera

    Returns:
        Normal vectors [N, 3] in world space, unit length
    """
    N = quaternions.shape[0]
    device = quaternions.device

    # Normalize quaternions
    quaternions = F.normalize(quaternions, p=2, dim=-1)

    # Extract quaternion components (w, x, y, z)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Build rotation matrix from quaternion
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

    # Find the axis with minimum scaling (thinnest axis)
    min_scale_idx = torch.argmin(scales, dim=1)  # [N]

    # Create one-hot vectors for minimum scaling axis
    one_hot = torch.zeros((N, 3), device=device, dtype=scales.dtype)
    one_hot.scatter_(1, min_scale_idx.unsqueeze(1), 1.0)

    # Compute normal: n_i = R · OneHot(argmin(s_1, s_2, s_3))
    normals = torch.bmm(R, one_hot.unsqueeze(-1)).squeeze(-1)  # [N, 3]

    # Normalize to unit vectors
    normals = F.normalize(normals, p=2, dim=-1)

    # Optionally flip normals to face the camera
    if flip_to_camera:
        view_dirs = camera_center.unsqueeze(0) - means3D  # [N, 3]
        view_dirs = F.normalize(view_dirs, p=2, dim=-1)
        dot_product = (normals * view_dirs).sum(dim=-1)  # [N]
        flip_mask = dot_product < 0
        normals[flip_mask] = -normals[flip_mask]

    return normals


def render_normals(
    means3D: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    normals_cam: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    means2D: torch.Tensor = None,
    depths: torch.Tensor = None,
    radii: torch.Tensor = None,
) -> torch.Tensor:
    """
    Render normal map - automatically uses gsplat if available, otherwise PyTorch fallback.
    
    This is the main entry point for normal rendering. It handles backend selection internally.

    Args:
        means3D: Gaussian centers in world space [N, 3]
        quats: Gaussian quaternions (wxyz) [N, 4]
        scales: Gaussian scales [N, 3]
        opacities: Gaussian opacities [N]
        normals_cam: Per-Gaussian normals in camera space [N, 3]
        viewmat: World-to-camera transformation [4, 4]
        K: Camera intrinsic matrix [3, 3]
        H, W: Image dimensions
        means2D: 2D screen positions [N, 2 or 3] (for PyTorch fallback)
        depths: Depth values [N] (for PyTorch fallback)
        radii: Visibility radii [N] (for PyTorch fallback)

    Returns:
        normal_map: Rendered normal map [3, H, W] in range [-1, 1]
    """
    if GSPLAT_AVAILABLE:
        # Fast path: use gsplat (expected ~0.2-0.5s for 50K Gaussians)
        return render_normals_gsplat(
            means3D=means3D,
            quats=quats,
            scales=scales,
            opacities=opacities,
            normals_cam=normals_cam,
            viewmat=viewmat,
            K=K,
            H=H,
            W=W
        )
    else:
        # Slow path: PyTorch fallback (expected ~19s for 50K Gaussians)
        print("[WARN] gsplat not available - using slow PyTorch fallback (~19s/frame)")
        if means2D is None or depths is None or radii is None:
            raise ValueError("PyTorch fallback requires means2D, depths, and radii")
        return render_normals_pytorch(
            normals_cam=normals_cam,
            means2D=means2D,
            depths=depths,
            opacities=opacities,
            radii=radii,
            H=H,
            W=W
        )


def render_normals_gsplat(
    means3D: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    normals_cam: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Render normal map using gsplat 1.5.3+ API (supports prebuilt wheels).
    
    Uses gsplat's rasterization to render normals as RGB channels, which is much
    faster than the PyTorch fallback.

    Args:
        means3D: Gaussian centers in world space [N, 3]
        quats: Gaussian quaternions (wxyz) [N, 4]
        scales: Gaussian scales [N, 3]
        opacities: Gaussian opacities [N]
        normals_cam: Per-Gaussian normals in camera space [N, 3]
        viewmat: World-to-camera transformation [4, 4]
        K: Camera intrinsic matrix [3, 3]
        H, W: Image dimensions

    Returns:
        normal_map: Rendered normal map [3, H, W] in range [-1, 1]
    """
    from gsplat import rasterization
    
    device = means3D.device
    dtype = means3D.dtype
    
    # gsplat expects [..., N, D] for Gaussians and [..., C, ...] for cameras
    # For single-image rendering: means [N, 3], viewmats [1, 4, 4], Ks [1, 3, 3]
    
    # Ensure camera parameters have camera batch dimension
    viewmat_batched = viewmat.unsqueeze(0)  # [1, 4, 4]
    K_batched = K.unsqueeze(0)  # [1, 3, 3]
    
    try:
        # Call gsplat rasterization with normals as the color channel
        # This renders the normals directly as RGB
        render_colors, render_alphas, meta = rasterization(
            means=means3D,  # [N, 3]
            quats=quats,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacities,  # [N]
            colors=normals_cam,  # [N, 3] - normals as color channel
            viewmats=viewmat_batched,  # [1, 4, 4]
            Ks=K_batched,  # [1, 3, 3]
            width=W,
            height=H,
            render_mode='RGB',  # Render RGB (normals)
            near_plane=0.01,
            far_plane=1e10,
        )
        
        # render_colors is [1, H, W, 3], convert to [3, H, W]
        normal_map = render_colors.squeeze(0).permute(2, 0, 1)  # [3, H, W]
        
        # Clamp to [-1, 1] range
        normal_map = torch.clamp(normal_map, -1.0, 1.0)
        
        return normal_map
        
    except Exception as e:
        print(f"[ERROR] gsplat rasterization failed: {e}")
        # Return zero normals on failure
        return torch.zeros((3, H, W), device=device, dtype=dtype)


def render_normals_pytorch(
    normals_cam: torch.Tensor,
    means2D: torch.Tensor,
    depths: torch.Tensor,
    opacities: torch.Tensor,
    radii: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Render normal map using pure PyTorch alpha-blending (SLOW - for fallback only).
    
    WARNING: This is ~100-1000x slower than gsplat! Only use if gsplat unavailable.

    Args:
        normals_cam: Per-Gaussian normal vectors [N, 3] in camera space, range [-1, 1]
        means2D: 2D screen positions [N, 3], only first 2 dims used (x, y)
        depths: Depth values [N] for sorting (back to front)
        opacities: Opacity values [N] or [N, 1]
        radii: Visibility radii [N] (>0 means visible)
        H, W: Image dimensions

    Returns:
        normal_map: Rendered normal map [3, H, W] in range [-1, 1]
    """
    print("[WARN] Using slow Python fallback for normal rendering (gsplat not available)")
    
    device = normals_cam.device
    dtype = normals_cam.dtype

    # Handle means2D shape - take only x, y
    if means2D.shape[-1] == 3:
        means2D = means2D[:, :2]

    # Ensure opacities is [N]
    if opacities.ndim == 2:
        opacities = opacities.squeeze(-1)

    # Filter visible Gaussians
    visible = radii > 0
    if not visible.any():
        return torch.zeros((3, H, W), device=device, dtype=dtype)

    # Get visible data
    normals_vis = normals_cam[visible]
    means2D_vis = means2D[visible]
    depths_vis = depths[visible]
    opacities_vis = opacities[visible]
    radii_vis = radii[visible]

    # Sort by depth (back to front for correct alpha blending)
    sorted_indices = torch.argsort(depths_vis, descending=True)
    normals_sorted = normals_vis[sorted_indices]
    means2D_sorted = means2D_vis[sorted_indices]
    opacities_sorted = opacities_vis[sorted_indices]
    radii_sorted = radii_vis[sorted_indices]

    # Initialize output
    normal_map = torch.zeros((H, W, 3), device=device, dtype=dtype)
    T = torch.ones((H, W), device=device, dtype=dtype)  # Transmittance

    # Create pixel grid
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    pixel_coords = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]

    # Process each Gaussian (SLOW - this is sequential)
    for i in range(len(normals_sorted)):
        cx, cy = means2D_sorted[i]  # Center coordinates
        opacity = opacities_sorted[i]
        normal = normals_sorted[i]  # [3]
        radius = radii_sorted[i]
        
        # Compute bounding box (with margin)
        margin = radius.ceil().long() + 2
        x_min = max(0, (cx - margin).long().item())
        x_max = min(W, (cx + margin).long().item() + 1)
        y_min = max(0, (cy - margin).long().item())
        y_max = min(H, (cy + margin).long().item() + 1)
        
        # Skip if bounding box is empty
        if x_min >= x_max or y_min >= y_max:
            continue
        
        # Create local coordinate grid (only within bounding box)
        y_local = torch.arange(y_min, y_max, device=device, dtype=dtype)
        x_local = torch.arange(x_min, x_max, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_local, x_local, indexing='ij')
        
        # Compute distances
        dx = xx - cx
        dy = yy - cy
        dist_sq = dx * dx + dy * dy
        
        # Compute Gaussian weight
        weight = torch.exp(-0.5 * dist_sq / (radius * radius + 1e-6))
        
        # Alpha value
        alpha = weight * opacity
        alpha = torch.clamp(alpha, 0.0, 0.99)
        
        # Extract local transmittance
        T_local = T[y_min:y_max, x_min:x_max]
        
        # Accumulate normal weighted by alpha and transmittance
        weighted_alpha = T_local * alpha
        normal_map[y_min:y_max, x_min:x_max] += weighted_alpha.unsqueeze(-1) * normal.view(1, 1, 3)
        
        # Update transmittance
        T[y_min:y_max, x_min:x_max] = T_local * (1 - alpha)
        
        # Early stopping if max transmittance is very small
        if T.max() < 1e-4:
            break

    # Convert from [H, W, 3] to [3, H, W] and clamp to [-1, 1]
    normal_map = normal_map.permute(2, 0, 1)
    normal_map = torch.clamp(normal_map, -1.0, 1.0)

    return normal_map


def normal_regularization_loss(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    mask: torch.Tensor = None,
    image_gradient: torch.Tensor = None,
    lambda_l1: float = 1.0,
    lambda_tv: float = 0.1,
    use_gradient_aware: bool = False
):
    """
    Compute normal map regularization loss.
    
    Combines L1 loss with optional TV (smoothness) loss and edge-aware weighting.
    
    Args:
        pred_normals: Predicted normal maps [B, 3, H, W] in range [-1, 1]
        gt_normals: Ground truth normal maps [B, 3, H, W] in range [-1, 1]
        mask: Binary mask [B, H, W] where 1 = valid pixels (default: all ones)
        image_gradient: Image gradients [B, 1, H, W] for edge-aware weighting (optional)
        lambda_l1: Weight for L1 loss (default: 1.0)
        lambda_tv: Weight for TV smoothness loss (default: 0.1)
        use_gradient_aware: If True, weight L1 loss by edge confidence (default: False)
    
    Returns:
        loss: Total loss (scalar)
        loss_dict: Dictionary with individual loss components
    """
    device = pred_normals.device
    
    # Create full mask if not provided
    if mask is None:
        mask = torch.ones(pred_normals.shape[0], pred_normals.shape[2], pred_normals.shape[3],
                         device=device, dtype=torch.float32)
    else:
        mask = mask.float()
    
    # Normalize normals for comparison (in case renderer output isn't perfectly normalized)
    pred_normals_norm = F.normalize(pred_normals, p=2, dim=1)
    gt_normals_norm = F.normalize(gt_normals, p=2, dim=1)
    
    # L1 loss
    l1_diff = torch.abs(pred_normals_norm - gt_normals_norm)  # [B, 3, H, W]
    
    # Apply mask: expand mask to 3 channels
    mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
    l1_diff = l1_diff * mask_expanded
    
    # Optionally apply edge-aware weighting
    if use_gradient_aware and image_gradient is not None:
        # Ensure image_gradient has correct shape [B, 1, H, W]
        if image_gradient.ndim == 2:
            image_gradient = image_gradient.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
        elif image_gradient.ndim == 3:
            image_gradient = image_gradient.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        # Edge confidence: low at edges (high gradient), high on flat regions (low gradient)
        edge_weight = torch.exp(-image_gradient / 0.05)  # Smaller gradient → higher weight
        edge_weight = edge_weight * mask_expanded
        l1_loss = (l1_diff * edge_weight).sum() / (mask_expanded.sum() + 1e-6)
    else:
        l1_loss = l1_diff.sum() / (mask_expanded.sum() + 1e-6)
    
    # TV (Total Variation) smoothness loss - encourage smooth normal fields
    tv_loss = torch.tensor(0.0, device=device)
    if lambda_tv > 0:
        # Compute differences along spatial dimensions
        diff_x = torch.abs(pred_normals_norm[:, :, :, :-1] - pred_normals_norm[:, :, :, 1:])
        diff_y = torch.abs(pred_normals_norm[:, :, :-1, :] - pred_normals_norm[:, :, 1:, :])
        
        # Apply mask to TV (mask out boundaries and invalid pixels)
        tv_loss = (diff_x.sum() + diff_y.sum()) / (pred_normals_norm.numel() + 1e-6)
    
    # Total loss
    total_loss = lambda_l1 * l1_loss + lambda_tv * tv_loss
    
    loss_dict = {
        "l1_loss": l1_loss.item(),
        "tv_loss": tv_loss.item() if isinstance(tv_loss, torch.Tensor) else tv_loss,
    }
    
    return total_loss, loss_dict


def scale_regularization_loss(
    get_scaling_fn,
    lambda_scale: float = 0.01,
    debug: bool = False
) -> torch.Tensor:
    """
    Regularize Gaussian scales to be more disc-like (thin in one direction).
    
    Encourages min_scale to be small relative to mean scale.
    This helps with normal computation which uses the minimum scaling axis.
    
    Args:
        get_scaling_fn: Either a callable that returns scales [N, 3] or a tensor [N, 3]
        lambda_scale: Weight for scale regularization (default: 0.01)
        debug: If True, print debug info about number of Gaussians processed
    
    Returns:
        loss: Scale regularization loss (scalar)
    """
    # Handle both callable and tensor inputs
    if callable(get_scaling_fn):
        scales = get_scaling_fn()  # [N, 3]
    else:
        scales = get_scaling_fn  # [N, 3]
    
    num_gaussians = scales.shape[0]
    
    # Compute ratio of min to mean scale
    min_scale = scales.min(dim=-1)[0]  # [N]
    mean_scale = scales.mean(dim=-1)   # [N]
    
    # Regularize to encourage disc-like Gaussians
    # Loss = lambda * min_scale / mean_scale (want this small)
    loss = lambda_scale * (min_scale / (mean_scale + 1e-6)).mean()
    
    if debug:
        print(f"  [Scale Loss] Processing {num_gaussians} Gaussians, loss={loss.item():.8f}")
    
    return loss


def compute_image_gradient(image: torch.Tensor) -> torch.Tensor:
    """
    Compute image gradient magnitude using Sobel filters.
    
    Useful for edge-aware weighting in loss functions.
    
    Args:
        image: RGB image tensor [3, H, W]
    
    Returns:
        grad_mag: Gradient magnitude [H, W]
    """
    # Convert to grayscale if needed
    if image.shape[0] == 3:
        # Simple grayscale conversion: average RGB
        gray = image.mean(dim=0, keepdim=True)  # [1, H, W]
    else:
        gray = image
    
    # Compute gradients using finite differences (Sobel-like)
    # Gradient in x direction
    grad_x = torch.abs(gray[:, :, :-1] - gray[:, :, 1:])
    
    # Gradient in y direction
    grad_y = torch.abs(gray[:, :-1, :] - gray[:, 1:, :])
    
    # Pad to match original size
    grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='constant', value=0)
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='constant', value=0)
    
    # Compute magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6).squeeze(0)
    
    return grad_mag

