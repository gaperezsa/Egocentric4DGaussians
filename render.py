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
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
from pathlib import Path
# import torch.multiprocessing as mp
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, aria):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    # breakpoint()
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.no_grad():
            if idx == 0:time1 = time()
            # breakpoint()
            rendering = to8b(render(view, gaussians, pipeline, background,cam_type=cam_type)["render"])
            torch.cuda.empty_cache()
            # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

            # Correct for arias innate rotation
            if aria:
                render_list.append(np.rot90(rendering.transpose(1,2,0),k=3,axes=(0,1)))
            else:
                render_list.append(rendering.transpose(1,2,0))
            # print(to8b(rendering).shape)
            if name in ["train", "test", "final_train_render"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                else:
                    gt  = view['image'].cuda()
                # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                gt_list.append(gt)
            # if idx >= 10:
                # break
    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))
    # print("writing training images.")

    multithread_write(gt_list, gts_path)
    # print("writing rendering images.")
    breakpoint()
    multithread_write(render_list, render_path)

    Path(os.path.join(model_path, name, "ours_{}".format(iteration))).mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)

def render_set_no_compression(model_path, name, iteration, views, gaussians, pipeline, background, cam_type, aria=False, override_color = None, override_opacity = None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_list = []
    gt_list = []

    print("point nums:", gaussians._xyz.shape[0])

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.no_grad():
            if idx == 0:
                time1 = time()

            # Render the current view, with no gradients required
            rendering = render(view, gaussians, pipeline, background, cam_type=cam_type, override_color = override_color, override_opacity = override_opacity)["render"]

            # Convert to float16 to save memory and move to CPU to free GPU memory
            rendering = rendering.half().cpu()

            # Correct for aria's innate rotation
            if aria:
                # Rotate the rendered image if aria correction is required
                rendering = rendering.permute(1, 2, 0)  # Convert to shape (H, W, C)
                rendering = torch.rot90(rendering, k=3, dims=(0, 1))  # Rotate 270 degrees (k=3)
                rendering = rendering.permute(2, 0, 1)  # Convert back to shape (C, H, W)

            # Append rendering to the CPU-based list
            render_list.append(rendering)

            if name in ["train", "test", "final_train_render", "color_by_movement"]:
                if cam_type != "PanopticSports":
                    gt = view.original_image[0:3, :, :]
                    if aria:
                        # Rotate the GT image if aria correction is required
                        gt = gt.permute(1, 2, 0)  # Convert to shape (H, W, C)
                        gt = torch.rot90(gt, k=3, dims=(0, 1))  # Rotate 270 degrees (k=3)
                        gt = gt.permute(2, 0, 1)  # Convert back to shape (C, H, W)

                else:
                    gt = view['image']
                    if aria:
                        # Rotate the GT image if aria correction is required
                        gt = gt.permute(1, 2, 0)  # Convert to shape (H, W, C)
                        gt = torch.rot90(gt, k=3, dims=(0, 1))  # Rotate 270 degrees (k=3)
                        gt = gt.permute(2, 0, 1)  # Convert back to shape (C, H, W)

                # Move GT to CPU and half precision
                gt = gt.half().cpu()

                # Append GT to the CPU-based list
                gt_list.append(gt)

            # Clear GPU memory
            torch.cuda.empty_cache()

    time2 = time()
    print("FPS:", (len(views) - 1) / (time2 - time1))

    # Save the images using multithreaded writing to speed up the process
    multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)

    # Create video from the saved images
    Path(os.path.join(model_path, name, "ours_{}".format(iteration))).mkdir(parents=True, exist_ok=True)
    render_images = [imageio.imread(os.path.join(render_path, f'{i:05d}.png')) for i in range(len(views))]
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool, aria: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type,aria)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type,aria)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type,aria)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--aria", action="store_true")
    parser.add_argument("--separate_depth_supervised", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.aria)