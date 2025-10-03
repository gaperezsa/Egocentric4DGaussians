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

def evaluate_single_folder_no_edges(folder_path: str, crop_px: int = 10, suffix: str = "no_edges"):
    """
    Evaluate metrics on a single folder after cropping 'crop_px' pixels
    from all four borders of both GT and render images.

    Input folder structure (same as evaluate_single_folder):
        folder_path/
          ├─ gt/
          └─ renders/

    Writes:
        folder_path/results_<suffix>.json
        folder_path/per_view_<suffix>.json
    """
    from pathlib import Path
    import os
    import json
    import numpy as np
    from PIL import Image
    import torch
    import torchvision.transforms.functional as tf
    from tqdm import tqdm
    from utils.loss_utils import ssim
    from utils.image_utils import psnr
    from lpipsPyTorch import lpips
    from pytorch_msssim import ms_ssim

    def _open_and_crop(img_path: Path, crop: int) -> Image.Image:
        img = Image.open(img_path)
        # Ensure 3-channel RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        # Effective crop cannot exceed half the dimension
        eff = min(crop, max((w - 1) // 2, 0), max((h - 1) // 2, 0))
        if eff > 0:
            box = (eff, eff, w - eff, h - eff)  # (left, top, right, bottom)
            img = img.crop(box)
        return img

    def _crop_tensor_2d(arr: torch.Tensor, crop: int) -> torch.Tensor:
        """
        Safely crop a 2D tensor [H,W] by 'crop' pixels on each side.
        If crop is 0 or leads to empty, it returns a minimally valid slice.
        """
        if crop <= 0:
            return arr
        H, W = arr.shape[-2], arr.shape[-1]
        eff = min(crop, max((W - 1) // 2, 0), max((H - 1) // 2, 0))
        if eff <= 0:
            return arr
        return arr[..., eff:H - eff, eff:W - eff]

    full_dict = {}
    per_view_dict = {}

    try:
        print(f"Evaluating folder (cropped {crop_px}px borders):", folder_path)

        gt_dir = Path(folder_path) / "gt"
        renders_dir = Path(folder_path) / "renders"

        if not gt_dir.exists() or not renders_dir.exists():
            raise FileNotFoundError(f"Expected subfolders 'gt' and 'renders' in {folder_path}")

        # Use the same filenames as in renders (as in your original code)
        image_names = sorted(os.listdir(renders_dir))

        # Prepare tensors
        renders = []
        gts = []
        for fname in image_names:
            r_img = _open_and_crop(renders_dir / fname, crop_px)
            g_img = _open_and_crop(gt_dir / fname, crop_px)
            # to [1,3,H,W] on CUDA to match your pipeline
            r_t = tf.to_tensor(r_img).unsqueeze(0)[:, :3, :, :].cuda()
            g_t = tf.to_tensor(g_img).unsqueeze(0)[:, :3, :, :].cuda()
            renders.append(r_t)
            gts.append(g_t)

        # Compute metrics
        ssims, psnrs, lpipss, lpipsa, ms_ssims, Dssims = [], [], [], [], [], []
        for idx in tqdm(range(len(renders)), desc=f"Metric evaluation (crop {crop_px}px)"):
            r = renders[idx]
            g = gts[idx]
            ssims.append(ssim(r, g))
            psnrs.append(psnr(r, g))
            lpipss.append(lpips(r, g, net_type='vgg'))
            ms_ssims.append(ms_ssim(r, g, data_range=1, size_average=True))
            lpipsa.append(lpips(r, g, net_type='alex'))
            Dssims.append((1 - ms_ssims[-1]) / 2)

        # Print results with a suffix tag
        tag = f"(no edges, crop={crop_px}px)"
        print("SSIM", tag, ": {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("PSNR", tag, ": {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("LPIPS-vgg", tag, ": {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        print("LPIPS-alex", tag, ": {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
        print("MS-SSIM", tag, ": {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
        print("D-SSIM", tag, ": {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

        # Store overall metrics
        full_dict.update({
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
            "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
            "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
            "D-SSIM": torch.tensor(Dssims).mean().item(),
            "crop_px": crop_px
        })

        # Per-image metrics
        per_view_dict.update({
            "SSIM": {name: v for v, name in zip(torch.tensor(ssims).tolist(), image_names)},
            "PSNR": {name: v for v, name in zip(torch.tensor(psnrs).tolist(), image_names)},
            "LPIPS-vgg": {name: v for v, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            "LPIPS-alex": {name: v for v, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
            "MS-SSIM": {name: v for v, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
            "D-SSIM": {name: v for v, name in zip(torch.tensor(Dssims).tolist(), image_names)},
            "crop_px": crop_px
        })

        # ---- Depth metrics (MSE / RMSE) comparable in spirit to EgoGaussian (without hand mask) ----
        depth_gt_dir = Path(folder_path) / "depth_gt_tensors"
        depth_renders_dir = Path(folder_path) / "depth_renders_tensors"

        depth_mses_list = []
        depth_rmses_list = []
        depth_mse_per_view = {}
        depth_rmse_per_view = {}

        if depth_gt_dir.exists() and depth_renders_dir.exists():
            # Match by filenames present in renders; look for .pt tensors
            depth_names = sorted([f for f in os.listdir(depth_renders_dir) if f.lower().endswith(".pt")])

            for fname in tqdm(depth_names, desc=f"Depth evaluation (crop {crop_px}px)"):
                pr_path = depth_renders_dir / fname
                gt_path = depth_gt_dir / fname
                if not gt_path.exists():
                    continue

                # Load tensors: expected shapes [H,W] (or any that can be squeezed to 2D)
                pr_t = torch.load(pr_path)
                gt_t = torch.load(gt_path)

                # Ensure float32
                pr_t = pr_t.to(torch.float32)
                gt_t = gt_t.to(torch.float32)

                # Squeeze any leading singular dims until 2D
                while pr_t.dim() > 2:
                    pr_t = pr_t.squeeze(0)
                while gt_t.dim() > 2:
                    gt_t = gt_t.squeeze(0)

                # Align shapes (crop to min H,W if needed)
                H = min(pr_t.shape[0], gt_t.shape[0])
                W = min(pr_t.shape[1], gt_t.shape[1])
                pr_t = pr_t[:H, :W]
                gt_t = gt_t[:H, :W]

                # Apply the same border crop as RGB
                pr_t = _crop_tensor_2d(pr_t, crop_px)
                gt_t = _crop_tensor_2d(gt_t, crop_px)

                # Valid mask: gt > 0 (as in EgoGaussian spirit)
                valid = (gt_t > 0).to(torch.float32)
                num_valid = valid.sum()
                if num_valid <= 0:
                    continue

                err = (pr_t - gt_t)
                mse = ((err ** 2) * valid).sum() / num_valid
                rmse = torch.sqrt(torch.clamp(mse, min=0.0))

                mse_val = float(mse.item())
                rmse_val = float(rmse.item())

                depth_mses_list.append(mse_val)
                depth_rmses_list.append(rmse_val)
                depth_mse_per_view[fname] = mse_val
                depth_rmse_per_view[fname] = rmse_val

            if depth_mses_list:
                depth_mse_mean = float(np.mean(depth_mses_list))
                depth_rmse_mean = float(np.mean(depth_rmses_list))
                print("DEPTH_MSE", tag, ": {:>12.7f}".format(depth_mse_mean, ".5"))
                print("DEPTH_RMSE", tag, ": {:>12.7f}".format(depth_rmse_mean, ".5"))

                # Add to overall dict
                full_dict.update({
                    "Depth MSE": depth_mse_mean,
                    "Depth RMSE": depth_rmse_mean
                })
                # Add per-view depth metrics
                per_view_dict.update({
                    "Depth MSE": depth_mse_per_view,
                    "Depth RMSE": depth_rmse_per_view
                })
            else:
                print("No valid depth pairs found for evaluation (after cropping / validity check).")
        else:
            print("Depth tensor directories not found; skipping depth MSE/RMSE.")

        # Save with suffix
        with open(str(Path(folder_path) / f"results_{suffix}.json"), 'w') as fp:
            json.dump(full_dict, fp, indent=True)
        with open(str(Path(folder_path) / f"per_view_{suffix}.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    except Exception as e:
        print("Unable to compute cropped-edge metrics for folder", folder_path)
        raise e


    
if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    #evaluate(args.model_paths)
    #evaluate_depth(args.model_paths)
    evaluate_single_folder_no_edges(args.model_paths[0], crop_px=10, suffix="no_edges_10px")
