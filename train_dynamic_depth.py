#
# train_dynamic_depth.py - main training entrypoint
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np  # numerical operations
import math
import random       # random sampling
import os, sys
import torch
from random import randint
from utils.exocentric_utils import compute_exocentric_from_file
from utils.graphics_utils import getWorld2View2
from utils.loss_utils import l1_loss, l1_filtered_loss, chamfer_loss, chamfer_with_median, l1_background_colored_masked_loss, gradient_aware_depth_loss, ssim
from utils.dn_splatter_utils import (
    normal_regularization_loss,
    scale_regularization_loss
)
from gaussian_renderer import render, network_gui, render_with_dynamic_gaussians_mask, render_dynamic_gaussians_mask_and_compare, get_deformed_gaussian_centers
from render import render_set_no_compression, render_all_splits
from metrics import evaluate_single_folder
from scene import Scene, GaussianModel, dynamics_by_depth
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image, debug_render_training_image_by_mask
from utils.render_utils import prune_by_visibility, prune_by_average_radius, look_at
from utils.metric_visualization_utils import generate_psnr_heatmaps_for_folder
from utils.pruning_utils import prune_invisible_dynamic_gaussians
from time import time
import copy
import json
import wandb
import socket
from datetime import datetime

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

def find_free_port(start_port=6000, max_attempts=100):
    """Find an available port on localhost"""
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"No free ports found in range {start_port}-{start_port+max_attempts}")

def generate_unique_expname():
    """Generate unique experiment name from W&B run or UUID"""
    try:
        if wandb.run is not None:
            return wandb.run.name
    except:
        pass
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]
    return f"sweep_{timestamp}_{run_id}"

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
# Ensure cameras are time-sorted
def check_and_sort_viewpoint_stack(viewpoint_stack):
    is_sorted = all(viewpoint_stack[i].time <= viewpoint_stack[i + 1].time for i in range(len(viewpoint_stack) - 1))
    if not is_sorted:
        viewpoint_stack.sort(key=lambda x: x.time)
    return viewpoint_stack

# Core training loop per stage
def dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, using_wandb, train_iter, timer, first_iter=0):

    # Setup learning rates based on stage
    gaussians.training_setup(opt, stage)

    #Initialize learning rates immediately to prevent gradient explosions at transition
    gaussians.initialize_learning_rates(iteration=0)
    
    # When transitioning between stages, the optimizer.zero_grad() from the last iteration
    gaussians.optimizer.zero_grad(set_to_none=True)

    rendering_only_background_or_only_dynamic = stage != "fine_coloring"
    depth_only_stage = stage in ("background_depth", "dynamics_depth")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    # Cache for per-camera ground truth point clouds (dynamics_depth optimization)
    gt_pointcloud_cache = {}

    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    # If not using DataLoader, load all train cams as a stack
    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = list(train_cams)
        viewpoint_stack = check_and_sort_viewpoint_stack(viewpoint_stack)
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("data loading done")
    
    # DataLoader branch
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, sampler=sampler, num_workers=16, collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    # zerostamp init branch 
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False

    count = 0

    # Main iteration loop
    for iteration in range(first_iter, final_iter + 1):
        # GUI handling
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                if custom_cam is not None:
                    count += 1
                    viewpoint_index = (count) % len(video_cams)
                    if (count // len(video_cams)) % 2 == 0:
                        pass
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    net_image = render_with_dynamic_gaussians_mask(
                        custom_cam, gaussians, pipe, background, scaling_modifier,
                        stage=stage, cam_type=scene.dataset_type
                    )["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, 0, 1) * 255).byte()
                                              .permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < train_iter) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Increase SH degree occasionally
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # DataLoader fetch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                if not random_loader:
                    print("reset dataloader into random dataloader.")
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size, shuffle=True, num_workers=16, collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
        else:
            # Manual sampling branch: random sampling without replacement
            idx = 0
            viewpoint_cams = []
            while idx < batch_size:
                if not viewpoint_stack:
                    if viewpoint_cams:
                        break
                    # Refill from full camera list
                    viewpoint_stack = temp_list.copy()
                # Sample a random view from the remaining stack
                rand_index = randint(0, len(viewpoint_stack) - 1)
                viewpoint_cam = viewpoint_stack.pop(rand_index)
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if not viewpoint_cams:
                continue

        # Debug mode after certain iteration
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # ====================================================================
        # STEP 1: Render and collect data for current batch
        # ====================================================================
        images, depth_images, dynamic_depth, dynamic_image = [], [], [], []
        dynamic_point_cloud, gt_images = [], []
        gt_depth_images, gt_dynamic_masks, gt_normal_maps = [], [], []
        gt_dynamic_point_cloud, radii_list = [], []
        visibility_filter_list, viewspace_point_tensor_list = [], []
        image_gradients = []  # Cache pre-computed image gradients
        rendered_normal_maps = []  # Rendered normals (only for normal loss stages)

        for viewpoint_cam in viewpoint_cams:
            viewpoint_cam.to_device("cuda")
            if viewpoint_cam.depth_image is None:
                continue

            # Special case: dynamics_depth stage only needs point clouds
            if stage == "dynamics_depth":
                dynamic_point_cloud.append(get_deformed_gaussian_centers(viewpoint_cam, gaussians))
                
                # Use cached backprojected point cloud (computed once per camera)
                cam_id = viewpoint_cam.uid
                if cam_id not in gt_pointcloud_cache:
                    gt_pointcloud_cache[cam_id] = viewpoint_cam.backproject_mask_to_world().squeeze().cuda()
                gt_dynamic_point_cloud.append(gt_pointcloud_cache[cam_id])
                continue
           
            # Render RGB and depth for this viewpoint
            # Determine if we need to render normals for this stage
            should_render_normals = (
                stage in ("background_depth", "background_RGB", "fine_coloring") and
                hyper.normal_loss_weight > 0 and
                viewpoint_cam.normal_map is not None
            )
           
            pkg = render_with_dynamic_gaussians_mask(
                viewpoint_cam, gaussians, pipe, background,
                stage=stage, cam_type=scene.dataset_type,
                training=rendering_only_background_or_only_dynamic,
                render_normals=should_render_normals  # NEW: flag-based normal rendering
            )
            
            image, depth_image = pkg["render"], pkg["depth"]
            viewspace_point_tensor, visibility_filter, radii = pkg["viewspace_points"], pkg["visibility_filter"], pkg["radii"]

            # Collect ground truth RGB (needed for gradient-aware losses)
            gt_image = viewpoint_cam.original_image if scene.dataset_type != "PanopticSports" else viewpoint_cam['image']
            gt_images.append(gt_image.unsqueeze(0))
            
            # Cache image gradient (computed once per camera, reused for gradient-aware losses)
            image_gradients.append(viewpoint_cam.get_image_gradient().unsqueeze(0))
            
            # Collect rendered RGB (only for RGB stages)
            if stage in ("background_RGB", "dynamics_RGB", "fine_coloring"):
                images.append(image.unsqueeze(0))

            # Always collect masks and depth
            gt_dynamic_masks.append(viewpoint_cam.dynamic_mask.unsqueeze(0))
            depth_images.append(depth_image.squeeze().unsqueeze(0))
            gt_depth_images.append(viewpoint_cam.depth_image.unsqueeze(0))
            
            # Collect ground truth and rendered normal maps (if available)
            if viewpoint_cam.normal_map is not None:
                gt_normal_maps.append(viewpoint_cam.normal_map.unsqueeze(0))
                
                # Get rendered normals from render package (computed if should_render_normals was True)
                if pkg["normal_map"] is not None:
                    rendered_normal_maps.append(pkg["normal_map"].unsqueeze(0))

            # Collect dynamic-specific renders (only for dynamics stages)
            if pkg.get("dynamic_only_render") is not None:
                dynamic_image.append(pkg["dynamic_only_render"].unsqueeze(0))
                dynamic_depth.append(pkg["dynamic_only_depth"].squeeze().unsqueeze(0))
                dynamic_point_cloud.append(pkg["dynamic_3D_means"].squeeze().cuda())
                
                # Use cached backprojected point cloud (computed once per camera)
                cam_id = viewpoint_cam.uid
                if cam_id not in gt_pointcloud_cache:
                    gt_pointcloud_cache[cam_id] = viewpoint_cam.backproject_mask_to_world().squeeze().cuda()
                gt_dynamic_point_cloud.append(gt_pointcloud_cache[cam_id])

            # Collect densification stats
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        
        # ====================================================================
        # STEP 2: Prepare tensors (concatenate batched data)
        # ====================================================================
        # Initialize all losses (unweighted, raw values)
        Ll1 = torch.tensor(0.0, device="cuda")
        depth_loss = torch.tensor(0.0, device="cuda")
        dynamic_mask_loss = torch.tensor(0.0, device="cuda")
        normal_loss = torch.tensor(0.0, device="cuda")
        scale_loss = torch.tensor(0.0, device="cuda")
        ssim_loss_val = torch.tensor(0.0, device="cuda")
        local_tv_loss = torch.tensor(0.0, device="cuda")
        median_dist = None  # Only computed in dynamics_RGB (for densification)

        # Concatenate common tensors (except for dynamics_depth stage)
        if stage != "dynamics_depth":
            gt_dynamic_masks_tensor = torch.cat(gt_dynamic_masks, 0)
            depth_image_tensor = torch.cat(depth_images, 0)
            gt_depth_image_tensor = torch.cat(gt_depth_images, 0)
            radii = torch.cat(radii_list, 0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        
        # Concatenate image gradient tensor if available
        image_gradient_tensor = None
        if hyper.use_gradient_aware_depth and len(image_gradients) > 0:
            image_gradient_tensor = torch.cat(image_gradients, 0)
        
        
        # ====================================================================
        # STEP 3: Compute losses per stage (UNWEIGHTED, organized by stage)
        # ====================================================================
        
        # ------------------ STAGE: background_depth ------------------
        if stage == "background_depth":
            # Depth loss: static regions only
            mask = ~gt_dynamic_masks_tensor
            if hyper.use_gradient_aware_depth and image_gradient_tensor is not None:
                depth_loss = gradient_aware_depth_loss(
                    depth_image_tensor, gt_depth_image_tensor,
                    image_gradient=image_gradient_tensor, mask=mask
                )
            else:
                # Standard L1 depth loss
                depth_loss = l1_filtered_loss(depth_image_tensor, gt_depth_image_tensor, mask, reduction="mean")
            
            # Normal loss: static regions only
            if len(rendered_normal_maps) > 0 and len(gt_normal_maps) > 0:
                rendered_normals_tensor = torch.cat(rendered_normal_maps, 0)
                gt_normals_tensor = torch.cat(gt_normal_maps, 0)
                normal_mask = ~gt_dynamic_masks_tensor  # Only static regions
                
                if hyper.use_gradient_aware_depth and image_gradient_tensor is not None:
                    normal_loss, _ = normal_regularization_loss(
                        pred_normals=rendered_normals_tensor,
                        gt_normals=gt_normals_tensor,
                        image_gradient=image_gradient_tensor,
                        mask=normal_mask,
                        lambda_l1=hyper.normal_l1_weight,
                        lambda_tv=hyper.normal_tv_weight,
                        use_gradient_aware=True
                    )
                else:
                    normal_loss, _ = normal_regularization_loss(
                        pred_normals=rendered_normals_tensor,
                        gt_normals=gt_normals_tensor,
                        mask=normal_mask,
                        lambda_l1=hyper.normal_l1_weight,
                        lambda_tv=hyper.normal_tv_weight,
                        use_gradient_aware=False
                    )
            
            # Scale regularization: encourage disc-like Gaussians
            if hyper.normal_loss_weight > 0:
                num_gaussians_before = gaussians.get_xyz.shape[0]
                scale_loss = scale_regularization_loss(lambda: gaussians.get_scaling, lambda_scale=0.01, debug=(iteration % 100 == 0))
        
        # ------------------ STAGE: background_RGB ------------------
        elif stage == "background_RGB":
            # RGB loss: static regions only
            image_tensor = torch.cat(images, 0)
            gt_image_tensor = torch.cat(gt_images, 0)
            mask = (~gt_dynamic_masks_tensor).unsqueeze(1).repeat(1, 3, 1, 1)
            Ll1 = l1_filtered_loss(image_tensor, gt_image_tensor[:, :3, :, :], mask)
            
            # Depth loss: static regions only
            depth_mask = ~gt_dynamic_masks_tensor
            if hyper.use_gradient_aware_depth and image_gradient_tensor is not None:
                depth_loss = gradient_aware_depth_loss(
                    depth_image_tensor, gt_depth_image_tensor,
                    image_gradient=image_gradient_tensor, mask=depth_mask
                )
            else:
                depth_loss = l1_filtered_loss(depth_image_tensor, gt_depth_image_tensor, depth_mask, reduction="mean")
            
            # Normal loss: static regions only
            if len(rendered_normal_maps) > 0 and len(gt_normal_maps) > 0:
                rendered_normals_tensor = torch.cat(rendered_normal_maps, 0)
                gt_normals_tensor = torch.cat(gt_normal_maps, 0)
                normal_mask = ~gt_dynamic_masks_tensor
                
                if hyper.use_gradient_aware_depth and image_gradient_tensor is not None:
                    normal_loss, _ = normal_regularization_loss(
                        pred_normals=rendered_normals_tensor,
                        gt_normals=gt_normals_tensor,
                        image_gradient=image_gradient_tensor,
                        mask=normal_mask,
                        lambda_l1=hyper.normal_l1_weight,
                        lambda_tv=hyper.normal_tv_weight,
                        use_gradient_aware=True
                    )
                else:
                    normal_loss, _ = normal_regularization_loss(
                        pred_normals=rendered_normals_tensor,
                        gt_normals=gt_normals_tensor,
                        mask=normal_mask,
                        lambda_l1=hyper.normal_l1_weight,
                        lambda_tv=hyper.normal_tv_weight,
                        use_gradient_aware=False
                    )
            
            # Scale regularization
            if hyper.normal_loss_weight > 0:
                scale_loss = scale_regularization_loss(lambda: gaussians.get_scaling, lambda_scale=0.01)
        
        # ------------------ STAGE: dynamics_depth ------------------
        elif stage == "dynamics_depth":
            # Chamfer loss on dynamic point clouds
            dynamic_mask_loss = chamfer_loss(dynamic_point_cloud, gt_dynamic_point_cloud)
        
        # ------------------ STAGE: dynamics_RGB ------------------
        elif stage == "dynamics_RGB":
            # RGB loss: dynamic regions only
            image_tensor = torch.cat(images, 0)
            gt_image_tensor = torch.cat(gt_images, 0)
            dynamic_image_tensor = torch.cat(dynamic_image, 0)
            mask = (gt_dynamic_masks_tensor).unsqueeze(1).repeat(1, 3, 1, 1)
            Ll1 = l1_background_colored_masked_loss(dynamic_image_tensor, gt_image_tensor[:, :3, :, :], mask, background)
            
            # Chamfer loss with median distance
            dynamic_mask_loss, median_dist = chamfer_with_median(dynamic_point_cloud, gt_dynamic_point_cloud)
            
            # Depth loss: dynamic regions only (masked GT depth)
            dynamic_only_depth_image_tensor = torch.cat(dynamic_depth, 0)
            zero_depth = torch.zeros_like(gt_depth_image_tensor)
            masked_gt_depth = torch.where(gt_dynamic_masks_tensor, gt_depth_image_tensor, zero_depth)
            
            if hyper.use_gradient_aware_depth and image_gradient_tensor is not None:
                depth_loss = gradient_aware_depth_loss(
                    dynamic_only_depth_image_tensor, masked_gt_depth,
                    image_gradient=image_gradient_tensor, mask=gt_dynamic_masks_tensor
                )
            else:
                depth_loss = l1_loss(dynamic_only_depth_image_tensor, masked_gt_depth)
            
            # Local space-time TV regularization
            if hyper.plane_tv_weight > 0:
                local_tv_loss = gaussians.compute_local_spacetime_tv()
        
        # ------------------ STAGE: fine_coloring ------------------
        elif stage == "fine_coloring":
            # RGB loss: full image
            image_tensor = torch.cat(images, 0)
            gt_image_tensor = torch.cat(gt_images, 0)
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
            
            # Depth loss: full image (with 0.5 weight factor)
            if hyper.use_gradient_aware_depth and image_gradient_tensor is not None:
                depth_loss = 0.5 * gradient_aware_depth_loss(
                    depth_image_tensor, gt_depth_image_tensor,
                    image_gradient=image_gradient_tensor, mask=None
                )
            else:
                depth_loss = 0.5 * l1_loss(depth_image_tensor, gt_depth_image_tensor)
            
            # Normal loss: static regions only
            if len(rendered_normal_maps) > 0 and len(gt_normal_maps) > 0:
                rendered_normals_tensor = torch.cat(rendered_normal_maps, 0)
                gt_normals_tensor = torch.cat(gt_normal_maps, 0)
                normal_mask = ~gt_dynamic_masks_tensor
                
                if hyper.use_gradient_aware_depth and image_gradient_tensor is not None:
                    normal_loss, _ = normal_regularization_loss(
                        pred_normals=rendered_normals_tensor,
                        gt_normals=gt_normals_tensor,
                        image_gradient=image_gradient_tensor,
                        mask=normal_mask,
                        lambda_l1=hyper.normal_l1_weight,
                        lambda_tv=hyper.normal_tv_weight,
                        use_gradient_aware=True
                    )
                else:
                    normal_loss, _ = normal_regularization_loss(
                        pred_normals=rendered_normals_tensor,
                        gt_normals=gt_normals_tensor,
                        mask=normal_mask,
                        lambda_l1=hyper.normal_l1_weight,
                        lambda_tv=hyper.normal_tv_weight,
                        use_gradient_aware=False
                    )
            
            # Scale regularization
            if hyper.normal_loss_weight > 0:
                scale_loss = scale_regularization_loss(lambda: gaussians.get_scaling, lambda_scale=0.01)
            
            # SSIM loss
            if hyper.ssim_weight != 0:
                ssim_loss_val = 1.0 - ssim(image_tensor, gt_image_tensor)
            
            # Local space-time TV regularization
            if hyper.plane_tv_weight > 0:
                local_tv_loss = gaussians.compute_local_spacetime_tv()
        
        # ====================================================================
        # STEP 4: Combine all losses with weights
        # ====================================================================

        loss = hyper.rgb_weight * Ll1 \
             + hyper.general_depth_weight * depth_loss \
             + hyper.chamfer_weight * dynamic_mask_loss \
             + hyper.normal_loss_weight * normal_loss \
             + hyper.scale_loss_weight * scale_loss \
             + hyper.ssim_weight * ssim_loss_val \
             + hyper.plane_tv_weight * local_tv_loss

        # ====================================================================
        # REGULARIZATION: Enforce zero deformation at t=0
        # ====================================================================
        # During dynamics_depth and dynamics_RGB stages, penalize any deformation at t=0
        # This forces the network to learn: at time 0, output ZERO displacement
        t0_reg = torch.tensor(0.0, device=Ll1.device)  # Initialize to zero
        t0_reg_weight = 1.0
        if stage in ("dynamics_depth", "dynamics_RGB") and hasattr(gaussians, '_deformation'):
            t0_reg_weight = 100.0  # Strength of the regularization
            t0_reg = gaussians._deformation.compute_t0_regularization(
                points=gaussians._xyz[gaussians._dynamic_xyz],
                scales=gaussians._scaling[gaussians._dynamic_xyz],
                rotations=gaussians._rotation[gaussians._dynamic_xyz],
                opacity=gaussians._opacity[gaussians._dynamic_xyz],
                shs=gaussians._features_dc[gaussians._dynamic_xyz],
                num_samples=min(2000, gaussians._dynamic_xyz.sum().item())
            )
            loss = loss + t0_reg_weight * t0_reg

        # ====================================================================
        # Log loss magnitudes and weighted contributions (every N iterations)
        # COMPREHENSIVE DEBUG: Show all losses for all stages (even if zero)
        # ====================================================================
        if iteration % 100 == 0:
            # Compute weighted contributions
            weighted_rgb = hyper.rgb_weight * Ll1.item()
            weighted_depth = hyper.general_depth_weight * depth_loss.item()
            weighted_chamfer = hyper.chamfer_weight * dynamic_mask_loss.item()
            weighted_normal = hyper.normal_loss_weight * normal_loss.item()
            weighted_scale = hyper.scale_loss_weight * scale_loss.item()
            weighted_ssim = hyper.ssim_weight * ssim_loss_val.item()
            weighted_tv = hyper.plane_tv_weight * local_tv_loss.item()
            weighted_t0_reg = t0_reg_weight * t0_reg.item()
            
            # Track which losses were actually computed (not just zero)
            depth_computed = (stage in ("background_depth", "background_RGB", "dynamics_RGB", "fine_coloring"))
            chamfer_computed = (stage in ("dynamics_depth", "dynamics_RGB"))
            normal_computed = (stage in ("background_depth", "background_RGB", "fine_coloring") and 
                              len(gt_normal_maps) > 0 and len(rendered_normal_maps) > 0)
            scale_computed = (stage in ("background_depth", "background_RGB", "fine_coloring") and 
                            hyper.normal_loss_weight > 0)
            ssim_computed = (stage == "fine_coloring" and hyper.ssim_weight != 0)
            tv_computed = (stage in ("dynamics_RGB", "fine_coloring") and hyper.plane_tv_weight > 0)
            t0_reg_computed = (stage in ("dynamics_depth", "dynamics_RGB") and hasattr(gaussians, '_deformation'))
            rgb_computed = (stage in ("background_RGB", "dynamics_RGB", "fine_coloring"))
            
            # Stage-specific sanity checks
            stage_info = {
                "background_depth": "Static depth only (RGB=0, Chamfer=0, SSIM=0)",
                "background_RGB": "Static RGB+depth only (Chamfer=0, SSIM=optional)",
                "dynamics_depth": "Dynamic point clouds only (RGB=0, Depth=optional, Normal=0, SSIM=0)",
                "dynamics_RGB": "Dynamic RGB+depth (Normal=optional)",
                "fine_coloring": "Full image refinement (all losses active)"
            }
            
            print(f"\n{'='*70}")
            print(f"[ITER {iteration:5d} | Stage: {stage:16s}] {stage_info.get(stage, '')}")
            print(f"{'='*70}")
            
            def format_loss_line(name, value, weight, weighted, computed, width_val=12):
                status = '✓' if (computed and value > 0) else ('✓ (zero)' if computed else '✗ NOT COMPUTED')
                return f"  {name:18s} {value:{width_val}.6f} (weight: {weight:6.4f}) → {weighted:{width_val}.6f}  {status}"
            
            print(format_loss_line("RGB Loss:", Ll1.item(), hyper.rgb_weight, weighted_rgb, rgb_computed))
            print(format_loss_line("Depth Loss:", depth_loss.item(), hyper.general_depth_weight, weighted_depth, depth_computed))
            print(format_loss_line("Chamfer Loss:", dynamic_mask_loss.item(), hyper.chamfer_weight, weighted_chamfer, chamfer_computed))
            print(format_loss_line("Normal Loss:", normal_loss.item(), hyper.normal_loss_weight, weighted_normal, normal_computed))
            print(format_loss_line("Scale Loss:", scale_loss.item(), hyper.scale_loss_weight, weighted_scale, scale_computed))
            print(format_loss_line("SSIM Loss:", ssim_loss_val.item(), hyper.ssim_weight, weighted_ssim, ssim_computed))
            print(format_loss_line("Spacetime TV:", local_tv_loss.item(), hyper.plane_tv_weight, weighted_tv, tv_computed))
            print(format_loss_line("T=0 Regularization:", t0_reg.item(), t0_reg_weight, weighted_t0_reg, t0_reg_computed))
            
            print(f"  {'-'*70}")
            print(f"  TOTAL LOSS:        {loss.item():12.6f}")
            print(f"{'='*70}")

        # ====================================================================
        # STEP 5: Compute PSNR (only for RGB stages)
        # ====================================================================
        psnr_ = 0
        if stage not in ("background_depth", "dynamics_depth"):
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        
        #if stage != "background_depth" and stage != "background_RGB" and hyper.time_smoothness_weight != 0:
        #    # tv_loss = 0
        #    tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
        #    loss += tv_loss

        if stage not in ("background_depth", "background_RGB") and hyper.plane_tv_weight > 0:
            # Our new “local space‐time TV” on just (XT, YT, ZT) rows that host dynamic Gaussians:
            local_tv = gaussians.compute_local_spacetime_tv()
            loss = loss + hyper.plane_tv_weight * local_tv

        if stage == "fine_coloring" and opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        loss.backward()

        #kill grad of no dynamic gaussians
        # if stage == "dynamics_depth":
        #    gaussians.erase_non_dynamic_grads()

        if stage != "dynamics_depth":
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                if viewspace_point_tensor_list[idx].grad is None:
                    continue
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, using_wandb, iteration, Ll1, loss, l1_loss, psnr_, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_with_dynamic_gaussians_mask, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 50 == 1) \
                    or (iteration < 3000 and iteration % 100 == 1) \
                        or (iteration < 60000 and iteration %  100 == 1) \
                            or (iteration < 200000 and iteration %  100 == 1) :
                            # breakpoint()
                            render_training_image(scene, gaussians, [train_cams[0]], render_with_dynamic_gaussians_mask, pipe, background, stage, iteration,timer.get_elapsed_time(),scene.dataset_type)
                            #render_training_image(scene, gaussians, [train_cams[500%len(train_cams)]], render_with_dynamic_gaussians_mask, pipe, background, stage+"_test_", iteration,timer.get_elapsed_time(),scene.dataset_type)
                            # Removed dyn_splat PNG output to save time and disk space
                            
                            # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                        # total_images.append(to8b(temp_image).transpose(1,2,0))

            timer.start()


            # Densification
            if (iteration > opt.densify_from_iter or iteration > opt.pruning_from_iter) and stage != "dynamics_depth":
                # Keep track of max radii in image-space for pruning
                # Keeping track of densification stats
                if stage == "dynamics_RGB":
                    combined_mask = gaussians._dynamic_xyz.clone()
                    combined_mask[gaussians._dynamic_xyz] = visibility_filter
                    gaussians.max_radii2D[combined_mask] = torch.max(gaussians.max_radii2D[combined_mask], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, combined_mask)
                else:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                # Defining thresholds for pruning and densifying
                if depth_only_stage:
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                elif stage == "fine_coloring":    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter ) 
                else:
                    opacity_threshold = opt.opacity_threshold_fine_after
                    densify_threshold = opt.densify_grad_threshold_after
                
                # Densify if in the middle of training
                if iteration > opt.densify_from_iter and iteration < opt.densify_until_iter and iteration < int(0.8 * final_iter) and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<400000:
                    
                    percentage_of_train_stage_remaining = 1-(iteration/final_iter)
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, median_dist, percentage_of_train_stage_remaining )
                
                # prune
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>150000:

                    #1) find & count outlier Gaussians on the last batch
                    gaussians.find_outlier_gaussians_and_update_hits(
                        viewpoint_cams,               # list of view cameras
                        depth_image_tensor,           # [B,H,W]
                        gt_depth_image_tensor,        # [B,H,W]
                        error_thresh_cm = 0.05,       # e.g. 0.05
                        topk_percent    = None,        # e.g. 0.10
                        search_radius   = None        # or some world-space radius
                    )

                    ## FOR PRUNE DEBUGGGING PURPOSES ##

                    # compute the mask *without* actually pruning:
                    mask = gaussians.compute_prune_mask(
                        opacity_threshold,
                        max_scale = opt.scale_pruning_factor * scene.cameras_extent,
                        max_screen_size = 0.5 * math.sqrt((viewpoint_cam.image_width**2) + (viewpoint_cam.image_height**2) ),
                        max_outlier_hits = 3  # e.g. 3
                    )

                    # Dump a debug image showing ONLY those Gaussians:
                    debug_render_training_image_by_mask(
                        scene, gaussians, [test_cams[0]],
                        render_with_dynamic_gaussians_mask,
                        mask, pipe, background,
                        stage, iteration,
                        timer.get_elapsed_time(), scene.dataset_type
                    )
                    ####################################


                    # 2) prune using all opacity + scale + image_size + depth outlier-hits
                    gaussians.prune(
                        opacity_threshold,
                        max_scale = opt.scale_pruning_factor * scene.cameras_extent,
                        max_screen_size = 0.5 * math.sqrt((viewpoint_cam.image_width**2) + (viewpoint_cam.image_height**2) ),
                        max_outlier_hits = 3  # e.g. 3
                    )
                    print(f"pruning in iter:{iteration}")
                    
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<450000 and opt.add_point and stage not in ("dynamics_RGB","fine_coloring", "background_depth"):
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    print(f"growing in iter:{iteration}")
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                    print(f"reseting opacity in iter:{iteration}")

            

            # Special pruning at end of dynamics_depth stage: Remove dynamic Gaussians not visible in canonical time
            if (stage == "dynamics_depth" and iteration == final_iter) or (stage == "dynamics_RGB" and iteration == first_iter+1):
                print("\n" + "="*80)
                print("[DYNAMICS_DEPTH FINAL PRUNING] Removing dynamic Gaussians invisible in canonical time")
                print("="*80)
                prune_invisible_dynamic_gaussians(
                    gaussians=gaussians,
                    scene=scene,
                    pipe=pipe,
                    background=background,
                    stage=stage,
                    num_cameras=5
                )
                print("="*80 + "\n")

                # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations) or iteration == final_iter:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")


def dynamic_depth_training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, using_wandb):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    cam_type=scene.dataset_type

    total_iters=opt.background_depth_iterations+opt.background_RGB_iterations+opt.dynamics_depth_iterations+opt.fine_iterations
    stages = ["background_depth", "background_RGB", "dynamics_depth", "dynamics_RGB", "fine_coloring"]
    first_iters=[0, 0, 0, 0, 0]
    training_iters=[opt.background_depth_iterations, opt.background_RGB_iterations, opt.dynamics_depth_iterations, opt.dynamics_RGB_iterations, opt.fine_iterations]
    current_stage = stages[0]
    
    if checkpoint:
        for i, st in enumerate(stages):
            if st in checkpoint.split("/")[-1]:
                (model_params, first_iter) = torch.load(checkpoint)
                gaussians.restore(model_params, opt, st)

                #Check the checkpoint iters is not far from the set iters, if so, just load and continue from the next stage
                if first_iter < training_iters[i] and abs(first_iter/(training_iters[i]+1)) < 0.9:
                    first_iters[i] = first_iter
                    current_stage = stages[i]
                elif i+1 >= len(stages):
                    current_stage = "final"
                    render_set_no_compression(dataset.model_path, stages[i] +"_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=False, render_func = render_with_dynamic_gaussians_mask, source_path=dataset.source_path, write_true_depth_gt=False)
                else:
                    current_stage = stages[i+1]
                    render_set_no_compression(dataset.model_path, stages[i] +"_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=False, render_func = render_with_dynamic_gaussians_mask, source_path=dataset.source_path, write_true_depth_gt=False)
                break
    

    if  current_stage == stages[0]: 
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "background_depth", tb_writer, using_wandb, training_iters[0], timer, first_iters[0])
        render_set_no_compression(dataset.model_path, "background_depth_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=False, render_func = render_with_dynamic_gaussians_mask, source_path=dataset.source_path, write_true_depth_gt=False)
        current_stage = stages[1]
    
    if  current_stage == stages[1]: 
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "background_RGB", tb_writer, using_wandb, training_iters[1], timer, first_iters[1])
        render_set_no_compression(dataset.model_path, "background_RGB_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=False, render_func = render_with_dynamic_gaussians_mask, source_path=dataset.source_path, write_true_depth_gt=False)
        current_stage = stages[2]

    
    if  current_stage == stages[2]:
        # Initialize dynamic gaussians in the model
        gaussians.spawn_dynamic_gaussians(random_init = False, precomputed_positions = list(scene.getTrainCameras())[1].backproject_mask_to_world())
        
        # FREEZE dynamic XYZ positions: We want the deformation network to learn temporal motion,
        # NOT to change the initialization. The spawn positions are perfect, so keep them fixed.
        if hasattr(gaussians, '_dynamic_xyz'):
            dynamic_mask = gaussians._dynamic_xyz
            gaussians._xyz.data[dynamic_mask].requires_grad = False
            print(f"✓ FROZEN dynamic Gaussian XYZ positions ({dynamic_mask.sum().item()} points)")
            print(f"  → Only deformation network will learn motion from t=0 position")
        
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "dynamics_depth", tb_writer, using_wandb, training_iters[2], timer, first_iters[2])
        render_set_no_compression(dataset.model_path, "dynamics_depth_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=False, render_func = render_with_dynamic_gaussians_mask, source_path=dataset.source_path, write_true_depth_gt=False)
        current_stage = stages[3]

    if  current_stage == stages[3]:
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "dynamics_RGB", tb_writer, using_wandb, training_iters[3], timer, first_iters[3])
        render_set_no_compression(dataset.model_path, "dynamics_RGB_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=False, render_func = render_with_dynamic_gaussians_mask, source_path=dataset.source_path, write_true_depth_gt=False)
        current_stage = stages[4]

    # No fine stage, just end the experiment
    if training_iters[4] == 0: return


    if  current_stage == stages[4]:
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "fine_coloring", tb_writer, using_wandb, training_iters[4], timer, first_iters[4])
        render_set_no_compression(dataset.model_path, "fine_coloring_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=False, render_func = render_with_dynamic_gaussians_mask, source_path=dataset.source_path, write_true_depth_gt=True)
    
    

    # —— visibility-based pruning just before final render ——
    '''
    
    print("Computing per-Gaussian visibility over all training frames…")
    prune_by_visibility(
        gaussians,
        scene.getTrainCameras(),
        render_with_dynamic_gaussians_mask,
        pipe,
        background,
        cam_type,
        threshold=0.2  # drop those seen in <20% of frames
    )
    
    print("Pruning by average screen‐radius…")
    prune_by_average_radius(
        gaussians,
        scene.getTrainCameras(),
        render_with_dynamic_gaussians_mask,
        pipe,
        background,
        cam_type,
        radius_thresh=20
    )
    '''


    #render_set_no_compression(dataset.model_path, "final_train_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
    #render_set_no_compression(dataset.model_path, "final_test_render", total_iters, scene.getTestCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
    
    '''
    # Show optinal exocentric camera render
    # —— static first‐frame view, offset “behind” by 10% —— 
    if scene.dataset_type == "colmap":
        orig_views = scene.getTrainCameras()
        if not orig_views:
            return

        # 1) Grab first frame’s transform & camera center
        v0 = orig_views[0]
        dev = v0.data_device
        M0 = v0.world_view_transform.to(dev)    # [4×4] world→view
        P0 = v0.projection_matrix.to(dev)       # [4×4] projection
        C0 = v0.camera_center.to(dev)           # [3]

        # 2) Compute forward‐world axis (camera looks along –Z in view space)
        R0 = M0[:3, :3]                         # the 3×3 rotation
        forward_world = -R0.transpose(0,1)[2]   # column 2 of R0ᵀ, negated
        forward_world = forward_world / forward_world.norm()

        # 3) Move back by, say, 10% of your scene’s extent
        dist_back = getattr(scene, "cameras_extent", 1.0) * 2
        C_back = C0 + forward_world * dist_back

        # 4) Compute new translation t = –R0 · C_back
        t_back = - (R0 @ C_back)

        # 5) Build the “behind” world→view matrix
        M_back = M0.clone()
        M_back[:3, 3] = t_back

        # 6) Apply the same M_back & P0 to *all* views
        static_views = []
        for v in orig_views:
            v.world_view_transform = M_back
            v.projection_matrix     = P0
            v.camera_center         = C_back
            v.full_proj_transform   = (
                M_back.unsqueeze(0)
                    .bmm(P0.unsqueeze(0))
                    .squeeze(0)
            )
            static_views.append(v)

        # 7) Render out
        render_set_no_compression(
            dataset.model_path,
            "exocentric_static_offset",
            total_iters,
            static_views,
            gaussians, pipe, background,
            cam_type=scene.dataset_type,
            aria=True,
            render_func=render_with_dynamic_gaussians_mask
        )
    '''

    # Visualize PSNR

    #final_train_folder = os.path.join(dataset.model_path, "final_train_render", f"ours_{total_iters}")
    #final_test_folder = os.path.join(dataset.model_path, "final_test_render", f"ours_{total_iters}")

    #generate_psnr_heatmaps_for_folder(
    #    final_train_folder,
    #    out_subdir="psnr_heatmaps",
    #    vmin=20.0,
    #    vmax=35.0,
    #    error_colors=True,
    #    make_video=True,
    #    fps=15,
    #    rotate_ccw_90=True,        # << rotate 90° CCW
    #    save_legend=True,          # writes psnr_colorbar_legend.png once
    #    add_colorbar_per_frame=False  # set True if you want the legend on every frame
    #)

    #evaluate_single_folder(os.path.join(dataset.model_path, "final_test_render", "ours_{}".format(total_iters)))

    # render all splits with original names
    render_all_splits(dataset.model_path, total_iters, scene, gaussians, pipe, background, cam_type, aria=False, render_func=render_with_dynamic_gaussians_mask, source_path=dataset.source_path)

    # evaluate each split folder (RGB+optional depth)
    split_root = os.path.join(dataset.model_path, "final_split_renders")
    for split_name in ["split_train", "split_test", "split_eval_static", "split_eval_dynamic"]:
        folder = os.path.join(split_root, split_name, f"ours_{total_iters}")
        if os.path.isdir(folder):
            try:
                evaluate_single_folder(folder)   # default crop_px=5 inside
            except Exception as e:
                print(f"[WARN] Metrics failed for {split_name} @ {folder}: {e}")

    # also evaluate the full sequence (optional)
    full_folder = os.path.join(dataset.model_path, "final_sequence_render", "sequence_full", f"ours_{total_iters}")
    if os.path.isdir(full_folder):
        try:
            evaluate_single_folder(full_folder)
        except Exception as e:
            print(f"[WARN] Metrics failed for sequence_full: {e}")

    

    
    
def prepare_output_and_logger(expname):    
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, using_wandb, iteration, Ll1, loss, l1_loss, psnr_, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
    if using_wandb:
        wandb.log({f'{stage}/train_loss_patches/l1_loss': Ll1.item(), f'{stage}/train_loss_patches/total_loss': loss.item(), f'{stage}/iter_time' : elapsed})
        wandb.log({f'{stage}/train_PSNR': psnr_,f'{stage}/iter_time' : elapsed})
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        if using_wandb and (idx < 5):
                            rendered_img = transforms.functional.to_pil_image(image[None], mode=None)
                            wandb.log({f"{stage}/{config['name']}_view_{viewpoint.image_name}/render": wandb.Image(rendered_img)})
                            if iteration == testing_iterations[0]:
                                gt_img = transforms.functional.to_pil_image(gt_image[None], mode=None)
                                wandb.log({f"{stage}/{config['name']}_view_{viewpoint.image_name}/ground_truth": wandb.Image(gt_img)})
                    except:
                        pass
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # mask=viewpoint.mask
                    
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                # print("sh feature",scene.gaussians.get_features.shape)
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if using_wandb:
                    wandb.log({f"{stage}/{config['name']}/loss_viewpoint - l1_loss": l1_test})
                    wandb.log({f"{stage}/{config['name']}/loss_viewpoint - psnr": psnr_test})
                    if stage in ("background_RGB", "dynamics_RGB") and psnr_test < 15:
                        print(f"tested in iteration {iteration} on stage {stage} and got a PSNR of {psnr_test}, stopping early in order to start next run ")
                        sys.exit(0)


        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1)/100, iteration,max_bins=500)
        if using_wandb:
            wandb.log({f'{stage}/scene/opacity_histogram': scene.gaussians.get_opacity})
            wandb.log({f'{stage}/total_points': scene.gaussians.get_xyz.shape[0]})
            wandb.log({f'{stage}/deformation_rate': scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0]})
            wandb.log({f'{stage}/scene/motion_histogram': scene.gaussians._deformation_accum.mean(dim=-1)/100})
        
        torch.cuda.empty_cache()
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def apply_sweep_overrides(op_params, hp_params, args):
    """
    Apply command-line argument overrides to parameter objects.
    This centralizes all parameter assignment logic in one place.
    
    Note: All parameters are already defined in OptimizationParams and ModelHiddenParams
    so they're automatically registered with argparse. This function only needs to be
    called if using additional custom arguments beyond those classes.
    """
    # Optional: Add any custom command-line only arguments here if needed
    # For now, all args are handled by the ParamGroup classes (OptimizationParams, ModelHiddenParams)
    return op_params, hp_params


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 1999, 4999, 7999, 9999])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 1999, 4999, 7999, 9999, 13900, 19999, 29999])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[ 1999, 4999, 7999, 9999, 13900, 19999, 29999])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--bounding_box_masked_depth_flag", action="store_true")
    parser.add_argument("--bs", type=int, default = 16)

    # ====================================================================
    # Grid-searched hyperparameters (Stage iterations)
    # ====================================================================
    parser.add_argument("--wandb", action="store_true")


    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.background_depth_iterations)
    args.save_iterations.append(args.background_RGB_iterations)
    args.save_iterations.append(args.dynamics_depth_iterations)
    args.save_iterations.append(args.dynamics_RGB_iterations)
    args.save_iterations.append(args.fine_iterations)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        # Pass sys.argv so CLI args take priority over config
        args = merge_hparams(args, config, cli_args=sys.argv)
    print("Optimizing " + args.model_path)

    # ====================================================================
    # ====================================================================
    # LOG TRAINING CONFIGURATION
    # ====================================================================
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    
    print("\n[Stage Iterations]")
    print(f"  background_depth: {args.background_depth_iterations}")
    print(f"  background_RGB: {args.background_RGB_iterations}")
    print(f"  dynamics_depth: {args.dynamics_depth_iterations}")
    print(f"  dynamics_RGB: {args.dynamics_RGB_iterations}")
    print(f"  fine_coloring: {args.fine_iterations}")
    
    print("\n[Loss Weights (from hyper)]")
    print(f"  rgb_weight: {args.rgb_weight}")
    print(f"  general_depth_weight: {args.general_depth_weight}")
    print(f"  chamfer_weight: {args.chamfer_weight}")
    print(f"  normal_loss_weight: {args.normal_loss_weight}")
    print(f"  scale_loss_weight: {args.scale_loss_weight}")
    print(f"  ssim_weight: {args.ssim_weight}")
    print(f"  scale_loss_weight: {args.scale_loss_weight}")
    print(f"  plane_tv_weight: {args.plane_tv_weight}")
    
    print("\n[Optimization Parameters]")
    print(f"  batch_size: {args.batch_size}")
    print(f"  pruning_interval: {args.pruning_interval}")
    print(f"  densification_interval: {args.densification_interval}")
    print(f"  static_position_lr: {args.static_position_lr_init:.0e} -> {args.static_position_lr_final:.0e}")
    print(f"  dynamic_position_lr: {args.dynamic_position_lr_init:.0e} -> {args.dynamic_position_lr_final:.0e}")
    print("="*70 + "\n")
    
    if args.wandb:
        print("intializing Weights and Biases...")
        wandb.init()
        config = wandb.config
        
        # Auto-assign port if set to 0
        if args.port == 0:
            args.port = find_free_port(6000)
            print(f"✓ Auto-assigned port: {args.port}")
        else:
            print(f"✓ Using specified port: {args.port}")
        
        # Generate the experiment name using the hyperparameters from the sweep
        experiment_name = generate_unique_expname()
        wandb.run.name = experiment_name
        args.expname = experiment_name
        print("Experiment Name:", experiment_name)

    # Initialize system state (RNG)
    #safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # ====================================================================
    # Extract and apply all parameters
    # ====================================================================
    op_params = op.extract(args)
    hp_params = hp.extract(args)
    
    # Handle multires conversion if needed
    if type(hp_params.multires) == type(list()) and type(hp_params.multires[0]) == type(str()):
        hp_params.multires = [int(x) for x in hp_params.multires]
    
    # Apply all command-line overrides in one clean function
    op_params, hp_params = apply_sweep_overrides(op_params, hp_params, args)
    
    # ====================================================================
    # Print parameter summary for debugging
    # ====================================================================
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"\n[Stage Iterations]")
    print(f"  background_depth: {op_params.background_depth_iterations}")
    print(f"  background_RGB: {op_params.background_RGB_iterations}")
    print(f"  dynamics_depth: {op_params.dynamics_depth_iterations}")
    print(f"  dynamics_RGB: {op_params.dynamics_RGB_iterations}")
    print(f"  fine_coloring: {op_params.fine_iterations}")
    
    print(f"\n[Loss Weights (from hyper)]")
    print(f"  rgb_weight: {hp_params.rgb_weight}")
    print(f"  general_depth_weight: {hp_params.general_depth_weight}")
    print(f"  chamfer_weight: {hp_params.chamfer_weight}")
    print(f"  normal_loss_weight: {hp_params.normal_loss_weight}")
    print(f"  scale_loss_weight: {hp_params.scale_loss_weight}")
    print(f"  ssim_weight: {hp_params.ssim_weight}")
    print(f"  scale_loss_weight: {hp_params.scale_loss_weight}")
    print(f"  plane_tv_weight: {hp_params.plane_tv_weight}")
    
    print(f"\n[Optimization Parameters]")
    print(f"  batch_size: {op_params.batch_size}")
    print(f"  pruning_interval: {op_params.pruning_interval}")
    print(f"  densification_interval: {op_params.densification_interval}")
    print(f"  static_position_lr: {op_params.static_position_lr_init} -> {op_params.static_position_lr_final}")
    print(f"  dynamic_position_lr: {op_params.dynamic_position_lr_init} -> {op_params.dynamic_position_lr_final}")
    print("="*70 + "\n")
    
    dynamic_depth_training(lp.extract(args), hp_params, op_params, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.wandb)

    # All done
    print("\nTraining complete.")
