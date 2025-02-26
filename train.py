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
import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss,l1_filtered_loss, l1_inverse_distance_loss, l1_proximity_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
from render import render_set_no_compression
from metrics import evaluate_single_folder
import sys
from scene import Scene, GaussianModel, dynamics_by_depth
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy
import json

import wandb

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
    
def check_and_sort_viewpoint_stack(viewpoint_stack):
    # Check if the list is already sorted
    is_sorted = all(viewpoint_stack[i].time <= viewpoint_stack[i + 1].time for i in range(len(viewpoint_stack) - 1))
    
    if not is_sorted:
        # Sort the list by the time attribute
        viewpoint_stack.sort(key=lambda x: x.time)
        
    return viewpoint_stack

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, using_wandb, train_iter,timer, first_iter=0):

    gaussians.training_setup(opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    # lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    #if stage == "close_dynamic":
    #    close_train_cams = scene.getCloseTrainCameras()
        
    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        viewpoint_stack = check_and_sort_viewpoint_stack(viewpoint_stack)
        temp_list = copy.deepcopy(viewpoint_stack)
        #filtered = [c for c in viewpoint_stack if c.depth_image is not None]
        
    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=16,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=16,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
                            # 
    count = 0

    # segmentation json
    #try:
    #    segmentation_file = open(dataset.source_path.replace(args.source_path.split("/")[-1], "instances.json"))
    #    segmentation_instances = json.load(segmentation_file)
    #    segmentation_instances['0'] = {'instance_id': 0, 'instance_name': 'empty', 'prototype_name': 'empty', 'category': 'nothing', 'category_uid': 0, 'motion_type': 'static', 'instance_type': 'human', 'rigidity': 'deformable', 'rotational_symmetry': {'is_annotated': False}, 'canonical_pose': {'up_vector': [0, 1, 0], 'front_vector': [0, 0, 1]}}
    #except:
    #    print("no segmentation instances file found")


    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count +=1
                    viewpoint_index = (count ) % len(video_cams)
                    if (count //(len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    # print(viewpoint_index)
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    # print(custom_cam.time, viewpoint_index, count)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(train_iter)) or not keep_alive) :
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)

        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                if not viewpoint_stack :
                    if len(viewpoint_cams) > 0:
                        break
                    else:
                        # Re initialize the stack from the begining
                        viewpoint_stack =  temp_list.copy()
                        
                viewpoint_cam = viewpoint_stack.pop(0)
                
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # print(len(viewpoint_cams))     
        # breakpoint()   
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        batch_dynamic_movement_loss = 0
        batch_out_of_frame_stillness_loss = 0
        images = []
        depth_images = []
        gt_images = []
        gt_depth_images = []
        gt_dynamic_static_masks = []
        gt_bounding_box_masks = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage,cam_type=scene.dataset_type, time_dynamic_loss=opt.time_dynamic_loss, out_of_frame_loss_flag=opt.out_of_frame_loss_flag)
            image, depth_image, viewspace_point_tensor, visibility_filter, radii, dynamic_movement_loss, out_of_frame_stillness_loss = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["dynamic_movement_loss"], render_pkg["out_of_frame_stillness_loss"]
            images.append(image.unsqueeze(0))
            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()

            # Does depth exist?
            if  viewpoint_cam.depth_image is not None and ( stage == "general_depth" or stage == "close_dynamic") :
                # Is depth valid?
                if viewpoint_cam.depth_image.max().item() < 1:
                    print("depth exists but its collapsed near the camera, ignoring this annotation")
                    gt_depth_image = None   
                else:
                    # Do we want to filter only depth outside the human bounding boxes?
                    if opt.bounding_box_masked_depth:
                        if viewpoint_cam.bounding_box_mask is not None:
                            bounding_box_mask = ~torch.tensor(viewpoint_cam.bounding_box_mask).cuda() #opposite of the bounding box for mask in depth filter
                            gt_bounding_box_masks.append(bounding_box_mask.unsqueeze(0))
                            depth_images.append(depth_image.squeeze().unsqueeze(0))
                            gt_depth_image = viewpoint_cam.depth_image.cuda()
                            gt_depth_images.append(gt_depth_image.unsqueeze(0))
                    else:
                        depth_images.append(depth_image.squeeze().unsqueeze(0))
                        gt_depth_image = viewpoint_cam.depth_image.cuda()
                        gt_depth_images.append(gt_depth_image.unsqueeze(0))
            else:
                gt_depth_image  = None
            
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            if dynamic_movement_loss is not None:
                batch_dynamic_movement_loss += dynamic_movement_loss
            if out_of_frame_stillness_loss is not None:
                batch_out_of_frame_stillness_loss += out_of_frame_stillness_loss
        
        # Take a reference state for out of frame loss in the next batch
        if "fine" in stage and opt.out_of_frame_loss_flag:
            gaussians.capture_reference_deformation_by_time(viewpoint_cam.time)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        
         # Loss
        # breakpoint()
        Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])
        Depth_loss = 0
        if len(gt_depth_images) > 0 and ( stage == "general_depth" or stage == "close_dynamic"):
            depth_image_tensor = torch.cat(depth_images,0)
            gt_depth_image_tensor = torch.cat(gt_depth_images,0)
            
            if stage == "general_depth":
                Depth_loss = l1_loss(depth_image_tensor, gt_depth_image_tensor)
                loss = Ll1 + (hyper.general_depth_weight * Depth_loss)
                
            elif stage == "close_dynamic":
                if opt.bounding_box_masked_depth:
                    gt_bounding_box_tensor = torch.cat(gt_bounding_box_masks,0)
                    #gt_dynamic_static_masks_tensor = torch.cat(gt_dynamic_static_masks,0)
                    Depth_loss = l1_filtered_loss(depth_image_tensor, gt_depth_image_tensor, gt_bounding_box_tensor)
                else:
                    Depth_loss = l1_filtered_loss(depth_image_tensor, gt_depth_image_tensor, depth_image_tensor < 1.0)
                loss = Ll1 + (hyper.close_dynamic_depth_weight * Depth_loss)
        else:
            loss = Ll1
        
        # Add new losses with weight hyperparameters
        # loss += hyper.dynamic_movement_weight*batch_dynamic_movement_loss + hyper.frame_stillness_weight*batch_out_of_frame_stillness_loss
        
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        
        if stage != "coarse" and hyper.time_smoothness_weight != 0:
            # tv_loss = 0
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor,gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        # if opt.lambda_lpips !=0:
        #     lpipsloss = lpips_loss(image_tensor,gt_image_tensor,lpips_model)
        #     loss += opt.lambda_lpips * lpipsloss
        
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        loss.backward()
        
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
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
            training_report(tb_writer, using_wandb, iteration, Ll1, loss, l1_loss, psnr_, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 50 == 1) \
                    or (iteration < 3000 and iteration % 100 == 99) \
                        or (iteration < 60000 and iteration %  500 == 499) \
                            or (iteration < 200000 and iteration %  1000 == 999) :
                            # breakpoint()
                            render_training_image(scene, gaussians, [test_cams[500%len(test_cams)]], render, pipe, background, stage+"_train_", iteration,timer.get_elapsed_time(),scene.dataset_type)
                            render_training_image(scene, gaussians, [train_cams[500%len(train_cams)]], render, pipe, background, stage+"_test_", iteration,timer.get_elapsed_time(),scene.dataset_type)
                            # render_training_image(scene, gaussians, train_cams, render, pipe, background, stage+"train", iteration,timer.get_elapsed_time(),scene.dataset_type)

                        # total_images.append(to8b(temp_image).transpose(1,2,0))
            timer.start()
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                elif stage == "fine":    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter ) 
                else:
                    opacity_threshold = opt.opacity_threshold_fine_after
                    densify_threshold = opt.densify_grad_threshold_after
                    
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
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
def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, using_wandb, training_mode=0):
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

    total_iters=opt.coarse_iterations+opt.iterations+opt.general_depth_iterations+opt.close_dynamic_iterations
    stages=["coarse","fine","general_depth","close_dynamic"]
    first_iters=[0,0,0,0]
    training_iters=[opt.coarse_iterations, opt.iterations, opt.general_depth_iterations, opt.close_dynamic_iterations]
    current_stage = stages[0]
    
    if checkpoint:
        for i, st in enumerate(stages):
            if st in checkpoint.split("/")[-1]:
                (model_params, first_iter) = torch.load(checkpoint)
                gaussians.restore(model_params, opt)
                #Check checkpoint iter is lower than the set training iters
                if first_iter > training_iters[i]:
                    training_iters[i] = first_iter

                #Check the checkpoint iters is not far from the set iters, if so, just load and continue from the next stage
                if abs(first_iter/(training_iters[i]+1) - 1) > 0.1:
                    current_stage = stages[i+1]
                else:
                    first_iters[i] = first_iter
                    current_stage = stages[i]
                break

    if (training_mode == 0 or training_mode == 1) and current_stage == "coarse": ## base training
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "coarse", tb_writer, using_wandb, opt.coarse_iterations, timer, first_iters[0])
        current_stage = "fine"

    if (training_mode == 0 or training_mode == 1) and current_stage == "fine": ## base training
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                                checkpoint_iterations, checkpoint, debug_from,
                                gaussians, scene, "fine", tb_writer, using_wandb, opt.iterations, timer, first_iters[1])
        current_stage = "general_depth"

    #if (training_mode == 1) and current_stage == "general_depth":   ## 4 stage training
    #    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
    #                            checkpoint_iterations, checkpoint, debug_from,
    #                            gaussians, scene, "general_depth", tb_writer, using_wandb, opt.general_depth_iterations, timer, first_iters[2])
    #    current_stage = "close_dynamic"

    distances, colors, opacities = dynamics_by_depth.movement_by_rendering(dataset.model_path, "dynamics", scene.getTrainCameras(), gaussians, pipe, background, cam_type)
    render_set_no_compression(dataset.model_path, "filtered_color_by_movement", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True, override_color = colors, override_opacity = opacities)
    return
    #if (training_mode == 1) and current_stage == "close_dynamic":   ## 4 stage training
    #    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
    #                            checkpoint_iterations, checkpoint, debug_from,
    #                            gaussians, scene, "close_dynamic", tb_writer, using_wandb, opt.close_dynamic_iterations, timer, first_iters[3])
    
    if training_mode == 2: ## probabilistic 2 stage
        raise NotImplementedError
    
    render_set_no_compression(dataset.model_path, "final_train_render", total_iters, scene.getTrainCameras(), gaussians, pipe, background, cam_type, aria=True)
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
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 9999, 14000, 20000, 45000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[ 40, 9999, 14000, 20000, 45000, 60000, 100000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[40, 9999, 14000, 20000, 45000, 60000, 100000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    parser.add_argument("--time_dynamic_loss_flag", action="store_true")
    parser.add_argument('--dynamic_movement_weight', type=float, default=0.5)
    parser.add_argument("--out_of_frame_loss_flag", action="store_true")
    parser.add_argument('--frame_stillness_weight', type=float, default=0.5)
    parser.add_argument("--bounding_box_masked_depth_flag", action="store_true")
    parser.add_argument('--general_depth_weight', type=float, default=0.5)
    parser.add_argument('--close_dynamic_depth_weight', type=float, default=1)
    
    #Which training pipeline, 0 is regular, 1 is with general depth and close depth, 2 is probabilistic paralel depth
    parser.add_argument('--training_mode', type=int, default=0)

    # grid_searched hyperparams
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--coarse_iter", type=int, default = 6000)
    parser.add_argument("--fine_iter", type=int, default = 20000)
    parser.add_argument("--general_depth_iter", type=int, default = 0)
    parser.add_argument("--close_dynamic_iter", type=int, default = 0)
    parser.add_argument("--netork_width", type=int, default = 128)
    parser.add_argument("--bs", type=int, default = 16)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    if args.wandb:
        print("intializing Weights and Biases...")
        wandb.init(
            # set the wandb project where this run will be logged
            project="4DGaussians",

            # track hyperparameters and run metadata
            config={
            "experiment_name" : args.expname,
            "coarse iterations": args.coarse_iter,
            "fine iterations": args.fine_iter,
            }
        )

    # Initialize system state (RNG)
    #safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Manual setting for sweeps
    op_params = op.extract(args)
    op_params.coarse_iterations =  args.coarse_iter
    op_params.iterations =  args.fine_iter
    op_params.general_depth_iterations =  args.general_depth_iter
    op_params.close_dynamic_iterations =  args.close_dynamic_iter
    op_params.batch_size =  args.bs
    op_params.time_dynamic_loss =  args.time_dynamic_loss_flag
    op_params.out_of_frame_loss_flag =  args.out_of_frame_loss_flag
    op_params.bounding_box_masked_depth = args.bounding_box_masked_depth_flag

    hp_params = hp.extract(args)
    hp_params.net_width =  args.netork_width
    hp_params.dynamic_movement_weight =  args.dynamic_movement_weight
    hp_params.frame_stillness_weight =  args.frame_stillness_weight
    hp_params.general_depth_weight = args.general_depth_weight
    hp_params.close_dynamic_depth_weight = args.close_dynamic_depth_weight
    

    training(lp.extract(args), hp_params, op_params, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname, args.wandb, args.training_mode)

    # All done
    print("\nTraining complete.")
