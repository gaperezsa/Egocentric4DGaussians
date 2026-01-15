#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from typing import Optional
import lpips
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_inverse_distance_loss(network_output, gt):
    return torch.abs((network_output - gt)/gt).mean()

def l1_proximity_loss(network_output, gt, max=1):
    close_filter = gt[gt<max]
    return torch.abs((network_output[close_filter] - gt[close_filter])).mean()

def l1_filtered_loss(network_output, gt, mask, reduction="mean"):
    if reduction == "mean":
        return torch.abs((network_output[mask] - gt[mask])).mean()
    elif reduction == "sum":
        return torch.abs((network_output[mask] - gt[mask])).sum()


def l1_background_colored_masked_loss(network_output, gt, mask, background_color):
    '''
    network_output of shape (bs,3,h,w)
    gt of shape (bs,3,h,w)
    mask of shape (bs,3,h,w)
    background_color of shape (3,)
    '''
    # Converting to tensor of broadcastable dimension and replacing outside the mask with background color
    backgrounded_image = torch.where(mask, gt, background_color.view(1,3,1,1))
    return torch.abs((network_output - backgrounded_image)).mean()

def l1_filtered_depth_valid_loss(network_output, gt, filter):
    valid_depth = gt > 0.001
    valid_filter = torch.logical_and(filter,valid_depth)
    return torch.abs((network_output[valid_filter] - gt[valid_filter])).mean()

def log_depth_loss(pred_depth, gt_depth, eps=1e-8):

    valid_depth = gt_depth > 0.001

    # Compute element-wise L1 distance between prediction and ground truth
    dist = torch.abs(pred_depth[valid_depth] - gt_depth[valid_depth])

    # Compute log(1 + L1 distance)
    log_term = torch.log(1.0 + dist + eps)

    # Average across all valid pixels
    loss = log_term.mean()

    return loss

def log_filtered_depth_loss(pred_depth, gt_depth, filter, eps=1e-8):

    valid_depth = gt_depth > 0.001
    valid_filter = torch.logical_and(filter,valid_depth)

    # Compute element-wise L1 distance between prediction and ground truth
    dist = torch.abs(pred_depth[valid_filter] - gt_depth[valid_filter])

    # Compute log(1 + L1 distance)
    log_term = torch.log(1.0 + dist + eps)

    # Average across all valid pixels
    loss = log_term.mean()

    return loss

def iou_loss(pred, gt, eps=1e-6):

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection + eps
    iou = intersection / union
    return 1 - iou

def recall_loss(pred, gt, eps=1e-6):

    intersection = (pred * gt).sum()
    recall = intersection / (gt.sum() + eps)
    return 1 - recall
    
import torch

def chamfer_loss(pred_list, gt_list, eps=1e-8, use_optimized=True):
    """
    Computes the average Chamfer distance over a batch.
    
    Each element in pred_list and gt_list is expected to be a torch.Tensor
    of shape (M, 3) and (N, 3) respectively, representing a set of 3D points.
    
    Args:
        pred_list (list of torch.Tensor): List of predicted 3D point sets.
        gt_list (list of torch.Tensor): List of ground-truth 3D point sets.
        eps (float): Small constant to avoid numerical issues.
        use_optimized (bool): Use optimized implementation (faster but slightly different numerical behavior).
        
    Returns:
        torch.Tensor: A scalar tensor containing the average Chamfer distance.
    """
    if use_optimized:
        return chamfer_loss_optimized(pred_list, gt_list, eps)
    else:
        return chamfer_loss_naive(pred_list, gt_list, eps)

def chamfer_loss_naive(pred_list, gt_list, eps=1e-8):
    """
    Original naive implementation: computes full O(N×M) pairwise distance matrix.
    Kept for reference and backward compatibility.
    """
    assert len(pred_list) == len(gt_list), "Batch sizes must match."
    batch_size = len(pred_list)
    chamfer_losses = []
    
    for pred_pts, gt_pts in zip(pred_list, gt_list):
        # Ensure the point sets have shape (N,3) (if not, reshape appropriately)
        if pred_pts.ndim != 2 or pred_pts.shape[1] != 3:
            raise ValueError("Each predicted point set must have shape (M, 3)")
        if gt_pts.ndim != 2 or gt_pts.shape[1] != 3:
            raise ValueError("Each ground-truth point set must have shape (N, 3)")
        
        # Handle empty point clouds (frames with no dynamic mask pixels)
        if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
            # Skip this frame - return 0 loss
            continue
        
        # Compute pairwise distances (squared) between predicted and ground-truth points.
        # Using torch.cdist for efficient pairwise distance computation.
        dist_matrix = torch.cdist(pred_pts, gt_pts, p=2).pow(2)  # shape (M, N)
        
        # For each predicted point, compute the min distance to any ground-truth point.
        min_dist_sq_pred, _ = torch.min(dist_matrix, dim=1)  # shape (M,)
        loss_pred = torch.sqrt(min_dist_sq_pred + eps).mean()
        
        # For each ground-truth point, compute the min distance to any predicted point.
        min_dist_sq_gt, _ = torch.min(dist_matrix, dim=0)  # shape (N,)
        loss_gt = torch.sqrt(min_dist_sq_gt + eps).mean()
        
        chamfer_losses.append(loss_pred + loss_gt)
    
    # Handle case where all frames had empty point clouds
    if len(chamfer_losses) == 0:
        return torch.tensor(0.0, device=pred_list[0].device if len(pred_list) > 0 else 'cuda')
    
    # Return the average Chamfer distance over the batch.
    return torch.stack(chamfer_losses).mean()

def chamfer_loss_optimized(pred_list, gt_list, eps=1e-8):
    """
    Optimized Chamfer distance using row-wise min operations (faster for many points).
    
    Instead of computing full O(N×M) matrix and then taking min, we compute
    distances row-by-row (or column-by-column) which is more memory efficient
    and can be faster for large point clouds.
    
    Args:
        pred_list (list of torch.Tensor): List of predicted 3D point sets.
        gt_list (list of torch.Tensor): List of ground-truth 3D point sets.
        eps (float): Small constant to avoid numerical issues.
        
    Returns:
        torch.Tensor: A scalar tensor containing the average Chamfer distance.
    """
    assert len(pred_list) == len(gt_list), "Batch sizes must match."
    batch_size = len(pred_list)
    chamfer_losses = []
    
    for pred_pts, gt_pts in zip(pred_list, gt_list):
        if pred_pts.ndim != 2 or pred_pts.shape[1] != 3:
            raise ValueError("Each predicted point set must have shape (M, 3)")
        if gt_pts.ndim != 2 or gt_pts.shape[1] != 3:
            raise ValueError("Each ground-truth point set must have shape (N, 3)")
        
        # Handle empty point clouds (frames with no dynamic mask pixels)
        if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
            # Skip this frame - return 0 loss
            continue
        
        # For each predicted point, find nearest ground-truth point
        # Compute squared L2 distance: ||pred - gt||^2 = sum((pred - gt)^2)
        # Using broadcasting: (M,1,3) - (1,N,3) -> (M,N,3) -> (M,N)
        pred_expanded = pred_pts.unsqueeze(1)  # (M, 1, 3)
        gt_expanded = gt_pts.unsqueeze(0)      # (1, N, 3)
        
        # Squared distances
        dist_sq = torch.sum((pred_expanded - gt_expanded) ** 2, dim=2)  # (M, N)
        
        # Min distance from each pred point to gt
        min_dist_sq_pred, _ = torch.min(dist_sq, dim=1)  # (M,)
        loss_pred = torch.sqrt(min_dist_sq_pred + eps).mean()
        
        # Min distance from each gt point to pred
        min_dist_sq_gt, _ = torch.min(dist_sq, dim=0)  # (N,)
        loss_gt = torch.sqrt(min_dist_sq_gt + eps).mean()
        
        chamfer_losses.append(loss_pred + loss_gt)
    
    # Handle case where all frames had empty point clouds
    if len(chamfer_losses) == 0:
        # Return zero loss (no dynamic content to supervise)
        return torch.tensor(0.0, device=pred_list[0].device if len(pred_list) > 0 else 'cuda')
    
    # Return the average Chamfer distance over the batch.
    return torch.stack(chamfer_losses).mean()

def chamfer_with_median(pred_list, gt_list, eps=1e-8, use_optimized=True):
    """
    Like chamfer_loss, but also collects all "pred→gt" nearest‐neighbor distances
    and returns their median.

    Args:
        use_optimized (bool): Use optimized implementation
        
    Returns:
        loss (Tensor scalar): average Chamfer distance over the batch.
        median_dist (float): median of sqrt(min_dist_sq_pred) over all pred points.
    """
    if use_optimized:
        return chamfer_with_median_optimized(pred_list, gt_list, eps)
    else:
        return chamfer_with_median_naive(pred_list, gt_list, eps)

def chamfer_with_median_naive(pred_list, gt_list, eps=1e-8):
    """
    Original implementation using torch.cdist (full O(N×M) matrix).
    """
    assert len(pred_list) == len(gt_list), "Batch sizes must match."
    chamfer_losses = []
    all_min_dists = []  # will store sqrt(min_dist_sq_pred) for every pred‐point

    for pred_pts, gt_pts in zip(pred_list, gt_list):
        # both must be [M,3] and [N,3]
        if pred_pts.ndim != 2 or pred_pts.shape[1] != 3:
            raise ValueError("Each predicted set must have shape (M,3)")
        if gt_pts.ndim != 2 or gt_pts.shape[1] != 3:
            raise ValueError("Each GT set must have shape (N,3)")

        # Handle empty point clouds (frames with no dynamic mask pixels)
        if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
            continue

        # 1) Compute squared pairwise distances [M×N]
        dist_matrix = torch.cdist(pred_pts, gt_pts, p=2).pow(2)  # [M,N]

        # 2) For each pred point, find its min‐squared‐dist to GT
        min_dist_sq_pred, _ = torch.min(dist_matrix, dim=1)  # [M]
        # record the *actual* (sqrt) distances
        all_min_dists.append(torch.sqrt(min_dist_sq_pred + eps))  # [M]

        # 3) For each GT point, find its min‐squared‐dist to pred
        min_dist_sq_gt, _ = torch.min(dist_matrix, dim=0)  # [N]

        # 4) per‐set Chamfer = mean over sqrt + mean over sqrt
        loss_pred = torch.sqrt(min_dist_sq_pred + eps).mean()
        loss_gt   = torch.sqrt(min_dist_sq_gt + eps).mean()
        chamfer_losses.append(loss_pred + loss_gt)

    # Handle empty case
    if len(chamfer_losses) == 0:
        return torch.tensor(0.0, device=pred_list[0].device if len(pred_list) > 0 else 'cuda'), 0.0

    # 5) Combine batch‐loss
    loss = torch.stack(chamfer_losses).mean()

    # 6) Flatten all "pred→gt" distances into one big vector, then median
    if len(all_min_dists) > 0:
        all_min_dists = torch.cat(all_min_dists, dim=0)  # length = total #pred‐points
        median_dist = torch.median(all_min_dists).item() # scalar float
    else:
        median_dist = 0.0

    return loss, median_dist

def chamfer_with_median_optimized(pred_list, gt_list, eps=1e-8):
    """
    Optimized version using row-wise operations instead of full matrix.
    """
    assert len(pred_list) == len(gt_list), "Batch sizes must match."
    chamfer_losses = []
    all_min_dists = []

    for pred_pts, gt_pts in zip(pred_list, gt_list):
        if pred_pts.ndim != 2 or pred_pts.shape[1] != 3:
            raise ValueError("Each predicted set must have shape (M,3)")
        if gt_pts.ndim != 2 or gt_pts.shape[1] != 3:
            raise ValueError("Each GT set must have shape (N,3)")

        # Handle empty point clouds (frames with no dynamic mask pixels)
        if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
            continue

        # Compute squared distances using broadcasting
        pred_expanded = pred_pts.unsqueeze(1)  # (M, 1, 3)
        gt_expanded = gt_pts.unsqueeze(0)      # (1, N, 3)
        dist_sq = torch.sum((pred_expanded - gt_expanded) ** 2, dim=2)  # (M, N)

        # For each pred point, find min distance to GT
        min_dist_sq_pred, _ = torch.min(dist_sq, dim=1)  # (M,)
        all_min_dists.append(torch.sqrt(min_dist_sq_pred + eps))  # (M,)

        # For each GT point, find min distance to pred
        min_dist_sq_gt, _ = torch.min(dist_sq, dim=0)  # (N,)

        # Per-set Chamfer
        loss_pred = torch.sqrt(min_dist_sq_pred + eps).mean()
        loss_gt   = torch.sqrt(min_dist_sq_gt + eps).mean()
        chamfer_losses.append(loss_pred + loss_gt)

    # Handle empty case
    if len(chamfer_losses) == 0:
        return torch.tensor(0.0, device=pred_list[0].device if len(pred_list) > 0 else 'cuda'), 0.0

    # Combine batch loss
    loss = torch.stack(chamfer_losses).mean()

    # Compute median
    if len(all_min_dists) > 0:
        all_min_dists = torch.cat(all_min_dists, dim=0)
        median_dist = torch.median(all_min_dists).item()
    else:
        median_dist = 0.0

    return loss, median_dist


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def dice_loss(y_pred, y_true, smooth=1e-6):
    # Convert boolean tensors to float
    y_pred = y_pred.float()
    y_true = y_true.float()
    
    # Calculate intersection and the sum of both masks
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum()
    
    # Compute Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice_coeff

def compute_image_gradient(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute image gradient magnitude using Sobel filters for DN-Splatter gradient-aware weighting.
    
    Args:
        image: RGB image tensor of shape (3, H, W) or (B, 3, H, W)
        eps: Small constant for numerical stability
        
    Returns:
        Gradient magnitude tensor of shape (H, W) or (B, H, W)
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Convert to grayscale (take mean across channels)
    if image.shape[1] == 3:
        gray = image.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    else:
        gray = image
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=image.device).view(1, 1, 3, 3)
    
    # Apply convolution
    grad_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    
    # Compute gradient magnitude
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    grad_mag = grad_mag.squeeze(1)  # Remove channel dimension
    
    if squeeze_output:
        grad_mag = grad_mag.squeeze(0)
    
    return grad_mag


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
    
    # Create valid depth mask
    valid_depth = gt_depth > depth_threshold
    
    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        valid_mask = torch.logical_and(valid_depth, mask)
    else:
        valid_mask = valid_depth
    
    # Compute L1 distance
    l1_dist = torch.abs(pred_depth - gt_depth)
    
    # Apply logarithmic penalty: log(1 + L1)
    log_term = torch.log(1.0 + l1_dist + eps)
    
    # Apply gradient-aware weighting
    weighted_loss = g_rgb * log_term
    
    # Average over valid pixels
    if valid_mask.any():
        loss = weighted_loss[valid_mask].mean()
    else:
        loss = torch.tensor(0.0, device=pred_depth.device)
    
    return loss
