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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def readDepthTensors(renders_dir, gt_dir):
    """
    Reads depth tensors from the specified directories.

    Args:
        renders_dir (Path): Path to the rendered depth tensors.
        gt_dir (Path): Path to the ground truth depth tensors.

    Returns:
        List[torch.Tensor]: List of rendered depth tensors.
        List[torch.Tensor]: List of ground truth depth tensors.
        List[str]: List of image names corresponding to each tensor.
    """
    renders = []
    gts = []
    image_names = []

    for fname in os.listdir(renders_dir):
        render = torch.load(renders_dir / fname).unsqueeze(0).cuda()
        gt = torch.load(gt_dir / fname).unsqueeze(0).cuda()
        renders.append(render)
        gts.append(gt)
        image_names.append(fname)

    return renders, gts, image_names

def mae(render, gt):
    """
    Computes Mean Absolute Error (MAE) between render and ground truth depth tensors.

    Args:
        render (torch.Tensor): Rendered depth tensor.
        gt (torch.Tensor): Ground truth depth tensor.

    Returns:
        float: MAE value.
    """
    valid_mask = gt > 0.001
    return torch.abs(render[valid_mask] - gt[valid_mask]).mean().item()

def rel_error(render, gt):
    """
    Computes Relative Error (Rel) between render and ground truth depth tensors,
    ignoring pixels where ground truth is zero or negative.

    Args:
        render (torch.Tensor): Rendered depth tensor.
        gt (torch.Tensor): Ground truth depth tensor.

    Returns:
        float: Rel value.
    """
    # Create a mask for valid ground truth values (gt > 0)
    valid_mask = gt > 0
    
    # Apply the mask to both tensors
    render_valid = render[valid_mask]
    gt_valid = gt[valid_mask]

    # Check if there are valid pixels left
    if gt_valid.numel() == 0:
        print("Warning: No valid depth pixels in ground truth!")
        return float('nan')

    # Compute Relative Error
    return (torch.abs(render_valid - gt_valid) / gt_valid).mean().item()

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                lpipsa = []
                ms_ssims = []
                Dssims = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    ms_ssims.append(ms_ssim(renders[idx], gts[idx],data_range=1, size_average=True ))
                    lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                    Dssims.append((1-ms_ssims[-1])/2)

                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                        "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                        "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                        "D-SSIM": torch.tensor(Dssims).mean().item()},

                                                    )
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                                            "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                                            "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},

                                                            }
                                                        )

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            
            print("Unable to compute metrics for model", scene_dir)
            raise e

def evaluate_depth(model_paths):
    """
    Evaluates depth tensors using MAE and Rel metrics.

    Args:
        model_paths (List[str]): List of model paths to evaluate.

    Writes:
        Results to `results.json` and `per_view.json`.
    """
    full_dict = {}
    per_view_dict = {}

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "depth_gt_tensors"
                renders_dir = method_dir / "depth_renders_tensors"

                renders, gts, image_names = readDepthTensors(renders_dir, gt_dir)

                maes = []
                rels = []

                for idx in tqdm(range(len(renders)), desc="Depth Metric Evaluation Progress"):
                    maes.append(mae(renders[idx], gts[idx]))
                    rels.append(rel_error(renders[idx], gts[idx]))

                print("Scene: ", scene_dir, "MAE : {:>12.7f}".format(torch.tensor(maes).mean(), ".5"))
                print("Scene: ", scene_dir, "Rel : {:>12.7f}".format(torch.tensor(rels).mean(), ".5"))

                full_dict[scene_dir][method].update({
                    "Depth MAE": torch.tensor(maes).mean().item(),
                    "Depth Rel": torch.tensor(rels).mean().item()
                })

                per_view_dict[scene_dir][method].update({
                    "Depth MAE": {name: mae for mae, name in zip(torch.tensor(maes).tolist(), image_names)},
                    "Depth Rel": {name: rel for rel, name in zip(torch.tensor(rels).tolist(), image_names)}
                })

            # Save results to JSON
            with open(scene_dir + "/results_depth.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view_depth.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)

        except Exception as e:
            print("Unable to compute depth metrics for model", scene_dir)
            raise e
        

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate_single_folder(folder_path):
    full_dict = {}
    per_view_dict = {}

    try:
        print("Evaluating folder:", folder_path)

        # Define the directories for GT and renders
        gt_dir = Path(folder_path) / "gt"
        renders_dir = Path(folder_path) / "renders"

        # Read all images
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        # Initialize lists to store all metrics
        ssims = []
        psnrs = []
        lpipss = []
        lpipsa = []
        ms_ssims = []
        Dssims = []

        # Evaluate metrics for all images
        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
            ms_ssims.append(ms_ssim(renders[idx], gts[idx], data_range=1, size_average=True))
            lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
            Dssims.append((1 - ms_ssims[-1]) / 2)

        # Print out the results
        print("SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
        print("MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
        print("D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

        # Store overall metrics
        full_dict.update({
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
            "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
            "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
            "D-SSIM": torch.tensor(Dssims).mean().item()
        })

        # Store per-image metrics
        per_view_dict.update({
            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
            "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
            "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)}
        })

        # Write results to files
        with open(folder_path + "/results.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(folder_path + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    except Exception as e:
        print("Unable to compute metrics for folder", folder_path)
        raise e
    
if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    #evaluate(args.model_paths)
    evaluate_depth(args.model_paths)
