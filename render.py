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
import os, re, json, imageio
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_with_dynamic_gaussians_mask as default_render
from utils.split_utils import load_splits_if_available, select_by_indices
from utils.depth_gt_utils import ensure_true_depth_gt
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
from pathlib import Path
from typing import List
import matplotlib as mpl

# import torch.multiprocessing as mp
import threading
import concurrent.futures

def multithread_save_tensors(tensor_list, path):
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    
    def save_tensor(tensor, count, path):
        try:
            save_path = os.path.join(path, '{0:05d}.pt'.format(count))
            torch.save(tensor, save_path)
            return count, True
        except Exception as e:
            print(f"Error saving tensor {count}: {e}")
            return count, False

    tasks = []
    for index, tensor in enumerate(tensor_list):
        tasks.append(executor.submit(save_tensor, tensor, index, path))
    
    executor.shutdown(wait=True)  # Wait for all threads to finish
    
    # Retry saving for failed tasks
    for task in tasks:
        index, success = task.result()
        if not success:
            save_tensor(tensor_list[index], index, path)

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
    multithread_write(render_list, render_path)

    Path(os.path.join(model_path, name, "ours_{}".format(iteration))).mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)

def _write_png_named(tensor_list, name_list, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for t, name in zip(tensor_list, name_list):
        p = os.path.join(out_dir, f"{name}.png")
        torchvision.utils.save_image(t.to(torch.float32), p)

def _save_tensors_named(tensor_list, name_list, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for t, name in zip(tensor_list, name_list):
        p = os.path.join(out_dir, f"{name}.pt")
        torch.save(t, p)

def _save_fixed_jet_np(depth_m_np: np.ndarray, out_png: str, vmin: float = 0.0, vmax: float = 2.5):
    """
    Save a jet heatmap with a FIXED metric range [vmin, vmax] meters:
      vmin (<=) -> blue, vmax (>=) -> red. Non-positive depth => black.
    """
    cmap = mpl.cm.get_cmap('jet')
    d = depth_m_np.copy()
    nonpos = d <= 0.0
    d = np.clip(d, vmin, vmax)
    denom = max(vmax - vmin, 1e-6)
    norm = (d - vmin) / denom
    vis = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    vis[nonpos] = 0
    imageio.imwrite(out_png, vis)

def _write_fixed_jet_from_tensors(run_dir: str, names: list, vmin: float = 0.0, vmax: float = 2.5):
    """
    Given a render run folder with:
      depth_renders_tensors/<name>.pt
      depth_gt_tensors/<name>.pt   (optional; created by ensure_true_depth_gt)
    create fixed-range jet PNGs in:
      depth_metric_vis_fixed/<name>.png      (pred)
      aligned_metric_vis_fixed/<name>.png    (compat alias to EgoGaussian)
      depth_gt_vis_fixed/<name>.png          (gt)
    """
    run = Path(run_dir)
    pred_pt_dir = run / "depth_renders_tensors"
    gt_pt_dir   = run / "depth_gt_tensors"

    pred_vis_dir = run / "depth_metric_vis_fixed"
    pred_vis_dir.mkdir(parents=True, exist_ok=True)
    # alias folder to look like EgoGaussian, if downstream expects it
    alias_vis_dir = run / "aligned_metric_vis_fixed"
    alias_vis_dir.mkdir(parents=True, exist_ok=True)

    gt_vis_dir = run / "depth_gt_vis_fixed"
    gt_vis_dir.mkdir(parents=True, exist_ok=True)

    for n in names:
        # predicted
        pt_p = pred_pt_dir / f"{n}.pt"
        if pt_p.is_file():
            d = torch.load(pt_p)
            while d.dim() > 2:
                d = d.squeeze(0)
            dnp = d.detach().cpu().numpy().astype(np.float32)
            _save_fixed_jet_np(dnp, str(pred_vis_dir / f"{n}.png"), vmin, vmax)
            _save_fixed_jet_np(dnp, str(alias_vis_dir / f"{n}.png"), vmin, vmax)

        # GT (true sparse)
        gtp = gt_pt_dir / f"{n}.pt"
        if gtp.is_file():
            g = torch.load(gtp)
            while g.dim() > 2:
                g = g.squeeze(0)
            gnp = g.detach().cpu().numpy().astype(np.float32)
            _save_fixed_jet_np(gnp, str(gt_vis_dir / f"{n}.png"), vmin, vmax)

@torch.no_grad()
def render_set_no_compression(
    model_path, name, iteration, views, gaussians, pipeline, background,
    cam_type, aria=False, override_color=None, override_opacity=None,
    render_func=default_render, subdir=None, use_original_filenames=True,
    make_videos=True, source_path=None, write_true_depth_gt=False
):
    """
    Renders a set of views to:
      <model_path>/<name>/<subdir or '.'>/ours_<iteration>/
        renders/                (RGB renders)          *.png
        gt/                     (RGB GT)               *.png
        depth_renders_tensors/  (pred depth tensors)   *.pt
        depth_renders/          (depth viz PNGs)       *.png
        # NO dense GT depth here
        # true sparse GT depth is optionally materialized from raw source:
        depth_gt_tensors/       (true GT, meters)      *.pt
        depth_gt/               (true GT viz)          *.png

    Filenames match view.image_name when use_original_filenames=True.
    If write_true_depth_gt=True and source_path is set, we populate true GT depth.
    """
    base_dir = os.path.join(model_path, name)
    if subdir is not None:
        base_dir = os.path.join(base_dir, subdir)
    run_dir = os.path.join(base_dir, f"ours_{iteration}")

    render_path = os.path.join(run_dir, "renders")
    gts_path = os.path.join(run_dir, "gt")
    depth_render_path = os.path.join(run_dir, "depth_renders")
    depth_render_tensors_path = os.path.join(run_dir, "depth_renders_tensors")

    for d in [render_path, gts_path, depth_render_path, depth_render_tensors_path]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("point nums:", gaussians._xyz.shape[0])

    def _filename_stem(view, fallback_idx: int) -> str:
        """
        Prefer trailing digits from view.image_name (drop leading zeros),
        else use the basename stem, else a zero-padded fallback index.
        """
        if use_original_filenames:
            raw = getattr(view, "image_name", None)
            if raw:
                base = os.path.splitext(os.path.basename(str(raw)))[0]
                m = re.search(r"(\d+)$", base)
                if m:
                    return str(int(m.group(1)))  # '00008' -> '8'
                return base                     # e.g. 'IMG_1234' → 'IMG_1234'
        return f"{fallback_idx:05d}"            # very last fallback

    # collections
    rgb_render_list, rgb_gt_list, names = [], [], []
    depth_render_vis_list, depth_render_tensor_list = [], []
    W = H = None
    first, t0 = True, 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if first:
            first, t0 = False, time()

        pkg = render_func(
            view, gaussians, pipeline, background,
            cam_type=cam_type,
            override_color=override_color,
            override_opacity=override_opacity
        )
        rendering = pkg["render"].detach()   # [3,H,W]
        depth     = pkg["depth"].detach()    # [1,H,W] or [H,W]

        # ---- aria fix (unchanged) ----
        if aria:
            rendering = rendering.permute(1, 2, 0)
            rendering = torch.rot90(rendering, k=3, dims=(0, 1))
            rendering = rendering.permute(2, 0, 1)
            d = depth.squeeze().cpu().numpy()
            d = np.rot90(d, k=3, axes=(0,1))
            d_vis = d / max(d.max(), 1e-6)
            depth_vis = torch.from_numpy(np.repeat(d_vis[..., None], 3, axis=2)).permute(2, 0, 1)
            depth_tensor = torch.from_numpy(d.copy())
        else:
            d = depth.squeeze().cpu()
            d_vis = d / max(d.max().item(), 1e-6)
            depth_vis = d_vis.unsqueeze(0).repeat(3, 1, 1)
            depth_tensor = d.cpu()

        if W is None or H is None:
            H, W = rendering.shape[-2], rendering.shape[-1]

        # collect tensors
        rgb_render_list.append(rendering.half().cpu())
        depth_render_vis_list.append(depth_vis.half())
        depth_render_tensor_list.append(depth_tensor)

        # RGB GT
        if cam_type != "PanopticSports":
            gt_rgb = view.original_image[0:3, :, :].detach()
            if aria:
                gt_rgb = gt_rgb.permute(1, 2, 0)
                gt_rgb = torch.rot90(gt_rgb, k=3, dims=(0, 1))
                gt_rgb = gt_rgb.permute(2, 0, 1)
            rgb_gt_list.append(gt_rgb.half().cpu())
        else:
            rgb_gt_list.append(view['image'].half().cpu())

        # final filename stem for this view
        names.append(_filename_stem(view, idx))

    if len(views) > 1:
        t1 = time()
        print("FPS:", (len(views) - 1) / max(t1 - t0, 1e-6))

    # ---- write using names ----
    _write_png_named(rgb_render_list, names, render_path)
    _write_png_named(rgb_gt_list, names, gts_path)
    _write_png_named(depth_render_vis_list, names, depth_render_path)
    _save_tensors_named(depth_render_tensor_list, names, depth_render_tensors_path)

    # manifest (unchanged shape)
    manifest = {
        "source_path": source_path,
        "image_names": names,  # these are the actual stems used on disk
        "width": W, "height": H,
        "cam_type": cam_type
    }
    with open(os.path.join(run_dir, "render_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # optionally materialize true GT depth now
    if write_true_depth_gt and source_path is not None:
        try:
            ensure_true_depth_gt(run_dir, source_path, names, (W, H))
        except Exception as e:
            print(f"[WARN] ensure_true_depth_gt failed: {e}")

    try:
        _write_fixed_jet_from_tensors(run_dir, names, vmin=0.0, vmax=2.5)
    except Exception as e:
        print(f"[WARN] fixed-range depth vis failed: {e}")

    # videos: sort numerically if possible, else lexicographically
    if make_videos:
        def _sort_names(ns):
            try:
                return sorted(ns, key=lambda s: int(s))
            except Exception:
                return sorted(ns)

        sorted_names = _sort_names(names)
        imgs = [imageio.imread(os.path.join(render_path, f"{n}.png")) for n in sorted_names]
        imageio.mimwrite(os.path.join(run_dir, 'video_rgb.mp4'), imgs, fps=15)
        dimgs = [imageio.imread(os.path.join(depth_render_path, f"{n}.png")) for n in sorted_names]
        imageio.mimwrite(os.path.join(run_dir, 'video_depth.mp4'), dimgs, fps=15)


def render_all_splits(
    model_path: str,
    iteration: int,
    scene,
    gaussians,
    pipeline,
    background,
    cam_type: str,
    render_func,                     # e.g. render_with_dynamic_gaussians_mask
    aria: bool = False,
    subroot: str = "final_split_renders",
    source_path: str = None,
):
    """
    Convenience wrapper: renders several subsets with original filenames:
      - sequence_full (all views)  [videos on]
      - split_train
      - split_test
      - split_eval_static
      - split_eval_dynamic
    """

    # 1) full sequence
    render_set_no_compression(
        model_path, "final_sequence_render", iteration,
        scene.getTrainCameras() + scene.getTestCameras(),
        gaussians, pipeline, background, cam_type, aria=aria,
        render_func=render_func, subdir="sequence_full",
        use_original_filenames=True, make_videos=True,
        source_path=source_path, write_true_depth_gt=True
    )

    split_cfg = load_splits_if_available(source_path or "")
    if split_cfg is None:
        print("[split render] no splits directory found; skipping per-split renders.")
        return

    # All frames ordered as in readColmapSceneInfo
    all_cams: List = list(scene.getTrainCameras()) + list(scene.getTestCameras())
    if not all_cams:
        all_cams = list(scene.getTrainCameras())

    def _do(indices: List[int], suffix: str):
        cams = select_by_indices(all_cams, indices)
        if not cams:
            print(f"[split render] split '{suffix}' is empty; skipping.")
            return
        # Use the new args your render function supports (subdir + true depth GT)
        render_set_no_compression(
            model_path,
            subroot,                 # parent name
            iteration,
            cams,
            gaussians,
            pipeline,
            background,
            cam_type,
            aria=aria,
            render_func=render_func,
            subdir=f"split_{suffix}",
            use_original_filenames=True,
            make_videos=False,
            source_path=source_path,
            write_true_depth_gt=True
        )

    _do(split_cfg["train"],        "train")
    _do(sorted(set(split_cfg["eval_static"]) | set(split_cfg["eval_dynamic"])), "test")
    _do(split_cfg["eval_static"],  "eval_static")
    _do(split_cfg["eval_dynamic"], "eval_dynamic")

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background,cam_type)
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
    parser.add_argument("--start_checkpoint", type=str, default = None)
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

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video, args.aria, args.start_checkpoint)