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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is not None:
        img1 = img1.flatten(1)
        img2 = img2.flatten(1)

        mask = mask.flatten(1).repeat(3,1)
        mask = torch.where(mask!=0,True,False)
        img1 = img1[mask]
        img2 = img2[mask]
        
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

    else:
        mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
    if mask is not None:
        if torch.isinf(psnr).any():
            print(mse.mean(),psnr.mean())
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse.float()))
            psnr = psnr[~torch.isinf(psnr)]
        
    return psnr

@torch.no_grad()
def psnr_map(img1: torch.Tensor,
             img2: torch.Tensor,
             mask: torch.Tensor = None,
             max_val: float = 1.0,
             eps: float = 1e-8,
             clamp_range: tuple=(20,40)):
    """
    Per-pixel PSNR map.

    Args:
        img1, img2: [B,3,H,W], range [0,1].
        mask: optional [B,1,H,W] or [B,H,W] boolean; False pixels will be NaN in the map.
        max_val: intensity max (1.0 for normalized tensors).
        eps: floor to avoid log(0).
        clamp_range: optional (vmin, vmax) to clamp PSNR map.

    Returns:
        psnr_map: [B,1,H,W] float32. Higher = better. NaN where mask=False.
    """
    # MSE per pixel across channels
    mse_map = (img1 - img2).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]
    # PSNR per pixel
    psnr_map = 10.0 * torch.log10((max_val * max_val) / torch.clamp(mse_map, min=eps))

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        psnr_map = psnr_map.masked_fill(~mask.bool(), float('nan'))

    if clamp_range is not None:
        vmin, vmax = clamp_range
        psnr_map = torch.clamp(psnr_map, vmin, vmax)

    return psnr_map