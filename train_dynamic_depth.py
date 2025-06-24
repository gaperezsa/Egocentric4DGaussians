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
import random       # random sampling
import os, sys
import torch
from random import randint
from utils.exocentric_utils import compute_exocentric_from_file
from utils.graphics_utils import getWorld2View2
from utils.loss_utils import l1_loss, l1_filtered_loss, chamfer_loss, chamfer_with_median, l1_background_colored_masked_loss
from gaussian_renderer import render, network_gui, render_with_dynamic_gaussians_mask, render_dynamic_gaussians_mask_and_compare, get_deformed_gaussian_centers
from render import render_set_no_compression
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
from utils.scene_utils import render_training_image
from utils.render_utils import prune_by_visibility, prune_by_average_radius, look_at
from time import time
import copy
import json
import wandb

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

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

    rendering_only_background_or_only_dynamic = stage != "fine_coloring"
    depth_only_stage = stage in ("background_depth", "dynamics_depth")
    hyper.general_depth_weight = 0.5  # temporary weight

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

        # Prepare batches of images, depths, etc.
        images, depth_images, dynamic_depth, dynamic_image = [], [], [], []
        dynamic_point_cloud, gt_images = [], []
        gt_depth_images, gt_dynamic_masks = [], []
        gt_dynamic_point_cloud, radii_list = [], []
        visibility_filter_list, viewspace_point_tensor_list = [], []
        median_dist = None

        for viewpoint_cam in viewpoint_cams:
            viewpoint_cam.to_device("cuda")
            if viewpoint_cam.depth_image is None:
                continue

            if stage == "dynamics_depth":
                dynamic_point_cloud.append(get_deformed_gaussian_centers(viewpoint_cam, gaussians))
                gt_dynamic_point_cloud.append(viewpoint_cam.backproject_mask_to_world().squeeze().cuda())
                continue
            pkg = render_with_dynamic_gaussians_mask(
                viewpoint_cam, gaussians, pipe, background,
                stage=stage, cam_type=scene.dataset_type,
                training=rendering_only_background_or_only_dynamic
            )
            image, depth_image = pkg["render"], pkg["depth"]
            viewspace_point_tensor, visibility_filter, radii = pkg["viewspace_points"], pkg["visibility_filter"], pkg["radii"]

            # Handle RGB ground truth
            if stage in ("background_RGB", "dynamics_RGB", "fine_coloring"):
                gt_image = viewpoint_cam.original_image if scene.dataset_type != "PanopticSports" else viewpoint_cam['image']
                images.append(image.unsqueeze(0)); gt_images.append(gt_image.unsqueeze(0))

            gt_dynamic_masks.append(viewpoint_cam.dynamic_mask.unsqueeze(0))
            depth_images.append(depth_image.squeeze().unsqueeze(0)); gt_depth_images.append(viewpoint_cam.depth_image.unsqueeze(0))

            if pkg.get("dynamic_only_render") is not None:
                dynamic_image.append(pkg["dynamic_only_render"].unsqueeze(0))
                dynamic_depth.append(pkg["dynamic_only_depth"].squeeze().unsqueeze(0))
                dynamic_point_cloud.append(pkg["dynamic_3D_means"].squeeze().cuda())
                gt_dynamic_point_cloud.append(viewpoint_cam.backproject_mask_to_world().squeeze().cuda())

            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        
        
        # Loss
        Ll1 = torch.tensor(0)
        depth_loss = torch.tensor(0)
        dynamic_mask_loss = torch.tensor(0)

        if stage != "dynamics_depth":
            gt_dynamic_masks_tensor = torch.cat(gt_dynamic_masks,0)

            depth_image_tensor = torch.cat(depth_images,0)
            gt_depth_image_tensor = torch.cat(gt_depth_images,0)

            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        
        # RGB loss
        if stage == "background_RGB" or stage == "dynamics_RGB" or stage == "fine_coloring":
            image_tensor = torch.cat(images,0)
            gt_image_tensor = torch.cat(gt_images,0)
            if stage == "background_RGB":
                mask = (~gt_dynamic_masks_tensor).unsqueeze(1).repeat(1, 3, 1, 1)
                Ll1 = l1_filtered_loss(image_tensor, gt_image_tensor[:,:3,:,:], mask)
            elif stage == "dynamics_RGB":
                mask = (gt_dynamic_masks_tensor).unsqueeze(1).repeat(1, 3, 1, 1)
                dynamic_image_tensor = torch.cat(dynamic_image,0)
                Ll1 = l1_background_colored_masked_loss(dynamic_image_tensor, gt_image_tensor[:,:3,:,:], mask, background)
            elif stage == "fine_coloring":
                Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

        # Depth and depth-related losses
        if stage == "background_depth":
            depth_loss = l1_filtered_loss(depth_image_tensor, gt_depth_image_tensor, ~gt_dynamic_masks_tensor)
        elif stage == "background_RGB":
            depth_loss = l1_filtered_loss(depth_image_tensor, gt_depth_image_tensor, ~gt_dynamic_masks_tensor)
            depth_loss = hyper.general_depth_weight * depth_loss
        elif stage == "dynamics_depth":
            dynamic_mask_loss = chamfer_loss(dynamic_point_cloud, gt_dynamic_point_cloud)
        elif stage == "dynamics_RGB":
            dynamic_only_depth_image_tensor = torch.cat(dynamic_depth,0)

            dynamic_mask_loss, median_dist = chamfer_with_median(dynamic_point_cloud, gt_dynamic_point_cloud)
            dynamic_mask_loss = opt.chamfer_weight * dynamic_mask_loss

            
            # gt_dynamic_masks_tensor has shape (bs, H, W), depth_image_tensor & gt_depth_image_tensor also (bs, H, W)
            zero_depth = torch.zeros_like(gt_depth_image_tensor)
            masked_gt_depth = torch.where(
                gt_dynamic_masks_tensor,        # inside dynamic mask
                gt_depth_image_tensor,          # keep true depth
                zero_depth                      # else set to 0
            )
            depth_loss = l1_loss(dynamic_only_depth_image_tensor, masked_gt_depth)
            depth_loss = hyper.general_depth_weight * depth_loss

        elif stage == "fine_coloring":
            depth_loss = l1_loss(depth_image_tensor, gt_depth_image_tensor)


        loss = Ll1 + depth_loss + dynamic_mask_loss

        psnr_ = 0
        if stage != "background_depth" and stage != "dynamics_depth":
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
                    or (iteration < 3000 and iteration % 100 == 99) \
                        or (iteration < 60000 and iteration %  1000 == 999) \
                            or (iteration < 200000 and iteration %  2000 == 1999) :
                            # breakpoint()
                            render_training_image(scene, gaussians, [test_cams[0%len(test_cams)]], render_with_dynamic_gaussians_mask, pipe, background, stage, iteration,timer.get_elapsed_time(),scene.dataset_type)
                            #render_training_image(scene, gaussians, [train_cams[500%len(train_cams)]], render_with_dynamic_gaussians_mask, pipe, background, stage+"_test_", iteration,timer.get_elapsed_time(),scene.dataset_type)
                            if stage == "dynamics_depth":
                                render_base_path = os.path.join(scene.model_path, f"{stage}_render")
                                image_path = os.path.join(render_base_path,"images")
                                render_pkg = render_with_dynamic_gaussians_mask(train_cams[0%len(train_cams)], gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type, training=True)
                                torchvision.utils.save_image((render_pkg["dynamic_only_depth"] > 0).detach().cpu().float(), render_base_path+f"_dyn_splat_{iteration}.png")

                            # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                        # total_images.append(to8b(temp_image).transpose(1,2,0))

            timer.start()


            # Densification
            if iteration < opt.densify_until_iter and stage != "dynamics_depth":
                # Keep track of max radii in image-space for pruning
                if stage == "dynamics_RGB":
                    combined_mask = gaussians._dynamic_xyz.clone()
                    combined_mask[gaussians._dynamic_xyz] = visibility_filter
                    gaussians.max_radii2D[combined_mask] = torch.max(gaussians.max_radii2D[combined_mask], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, combined_mask)
                else:
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if depth_only_stage:
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                elif stage == "fine_coloring":    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter ) 
                else:
                    opacity_threshold = opt.opacity_threshold_fine_after
                    densify_threshold = opt.densify_grad_threshold_after
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<450000 and stage in ("dynamics_depth", "dynamics_RGB"):
                    gaussians.densify_dynamic(densify_threshold, opacity_threshold, scene.cameras_extent, median_dist)
                elif iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<450000:
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, median_dist)
                
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<450000 and opt.add_point and stage not in ("dynamics_RGB","fine_coloring"):
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
                    
            

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
                gaussians.restore(model_params, opt)

                #Check the checkpoint iters is not far from the set iters, if so, just load and continue from the next stage
                if first_iter < training_iters[i] and abs(first_iter/(training_iters[i]+1)) < 0.9:
                    first_iters[i] = first_iter
                    current_stage = stages[i]
                elif i+1 >= len(stages):
                    current_stage = "final"
                    render_set_no_compression(dataset.model_path, stages[i] +"_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
                else:
                    current_stage = stages[i+1]
                    render_set_no_compression(dataset.model_path, stages[i] +"_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
                break
    
    

    if  current_stage == stages[0]: 
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "background_depth", tb_writer, using_wandb, training_iters[0], timer, first_iters[0])
        render_set_no_compression(dataset.model_path, "background_depth_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
        current_stage = stages[1]
    
    if  current_stage == stages[1]: 
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "background_RGB", tb_writer, using_wandb, training_iters[1], timer, first_iters[1])
        render_set_no_compression(dataset.model_path, "background_RGB_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
        current_stage = stages[2]
        
    if  current_stage == stages[2]:
        # Initialize dynamic gaussians in the model
        gaussians.spawn_dynamic_gaussians()
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "dynamics_depth", tb_writer, using_wandb, training_iters[2], timer, first_iters[2])
        render_set_no_compression(dataset.model_path, "dynamics_depth_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
        current_stage = stages[3]

    if  current_stage == stages[3]:
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "dynamics_RGB", tb_writer, using_wandb, training_iters[3], timer, first_iters[3])
        render_set_no_compression(dataset.model_path, "dynamics_RGB_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
        current_stage = stages[4]
    


    if  current_stage == stages[4]:
        dynamic_depth_scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "fine_coloring", tb_writer, using_wandb, training_iters[4], timer, first_iters[4])
        render_set_no_compression(dataset.model_path, "fine_coloring_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
    
    

    # —— visibility-based pruning just before final render ——
    
    
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
    '''
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


    render_set_no_compression(dataset.model_path, "final_train_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, render_func = render_with_dynamic_gaussians_mask)
    
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




    evaluate_single_folder(os.path.join(dataset.model_path, "final_train_render", "ours_{}".format(total_iters)))
    
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 9999, 13999, 19999, 29999])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 999, 4999, 7999, 9999, 13900, 19999, 29999])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[ 999, 4999, 9999, 13900, 19999, 29999])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--bounding_box_masked_depth_flag", action="store_true")
    parser.add_argument('--chamfer_weight', type=float, default=50.0)
    parser.add_argument('--fine_opt_dyn_lr_downscaler', type=float, default=0.01)

    # grid_searched hyperparams
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--background_depth_iter", type=int, default = 20000)
    parser.add_argument("--background_RGB_iter", type=int, default = 14000)
    parser.add_argument("--dynamics_depth_iter", type=int, default = 10000)
    parser.add_argument("--dynamics_RGB_iter", type=int, default = 5000)
    parser.add_argument("--fine_iter", type=int, default = 20000)
    parser.add_argument("--netork_width", type=int, default = 128)
    parser.add_argument("--bs", type=int, default = 16)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.background_depth_iter)
    args.save_iterations.append(args.background_RGB_iter)
    args.save_iterations.append(args.dynamics_depth_iter)
    args.save_iterations.append(args.dynamics_RGB_iter)
    args.save_iterations.append(args.fine_iter)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    if args.wandb:
        print("intializing Weights and Biases...")
        wandb.init()
        config = wandb.config
        # Define your experiment name template (you can also hardcode it here or pass it via the config)
        #name_template = "exp_bd{background_depth_iter}_dd{dynamics_depth_iter}_defor{defor_depth}_width{net_width}_gridlr{grid_lr_init}"
        name_template = "BASELINE_{video_number}_backDepth{background_depth_iter}_backRGB{background_RGB_iter}_dynDepth{dynamics_depth_iter}_dynRGB{dynamics_RGB_iter}_fine{fine_iter}_chamferWeight{chamfer_weight}_fineOptDynLrDownscaler{fine_opt_dyn_lr_downscaler}_colouredBackground"

        # Generate the experiment name using the hyperparameters from the sweep
        experiment_name = name_template.format(
            video_number = config.source_path.split('/')[-2],
            background_depth_iter = config.background_depth_iter,
            background_RGB_iter = config.background_RGB_iter,
            dynamics_depth_iter = config.dynamics_depth_iter,
            dynamics_RGB_iter = config.dynamics_RGB_iter,
            fine_iter = config.fine_iter,
            chamfer_weight = config.chamfer_weight,
            fine_opt_dyn_lr_downscaler = config.fine_opt_dyn_lr_downscaler
        )

        # Optionally, update the run name in wandb
        wandb.run.name = experiment_name

        args.expname = experiment_name

        print("Experiment Name:", experiment_name)

    # Initialize system state (RNG)
    #safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Manual setting for sweeps
    op_params = op.extract(args)
    op_params.background_depth_iterations =  args.background_depth_iter
    op_params.background_RGB_iterations =  args.background_RGB_iter
    op_params.dynamics_depth_iterations =  args.dynamics_depth_iter
    op_params.dynamics_RGB_iterations =  args.dynamics_RGB_iter
    op_params.fine_iterations =  args.fine_iter
    op_params.batch_size =  args.bs
    op_params.chamfer_weight = args.chamfer_weight
    op_params.fine_opt_dyn_lr_downscaler = args.fine_opt_dyn_lr_downscaler

    hp_params = hp.extract(args)
    if type(hp_params.multires) == type(list()) and type(hp_params.multires[0]) == type(str()):
        hp_params.multires = [int(x) for x in hp_params.multires]
    hp_params.net_width =  args.netork_width
    

    dynamic_depth_training(lp.extract(args), hp_params, op_params, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.wandb)

    # All done
    print("\nTraining complete.")
