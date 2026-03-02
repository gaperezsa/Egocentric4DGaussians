"""
utils/importance_sampling_utils.py
-----------------------------------
Utilities for per-view importance sampling during the fine_coloring stage.

The idea:
  Right after dynamics_RGB finishes, render ALL training views once (no_grad),
  compute per-frame RGB L1 loss, and flag the frames whose loss exceeds
  mean + threshold_sigma * std as "high-loss".  Before fine_coloring begins,
  those cameras are duplicated in the sampling pool so they are seen ~2x as
  often during training.

Public API (used by train_dynamic_depth.py):
  - compute_high_loss_cam_ids(...)  -> set[int]
  - build_importance_sampled_pool(...)  -> list[Camera]
"""

import os
import copy
import numpy as np
import torch
import cv2


def compute_high_loss_cam_ids(scene, gaussians, render_fn, pipe, background,
                               cam_type, model_path, threshold_sigma=1.0):
    """Render every training camera and return UIDs of frames whose RGB L1 loss
    exceeds  mean + threshold_sigma * std.

    Saves side-by-side rendered/GT comparison images for every high-loss frame
    to ``model_path/high_loss_training_views/`` so they can be inspected before
    fine_coloring starts.

    Parameters
    ----------
    scene       : Scene object with .getTrainCameras()
    gaussians   : current GaussianModel
    render_fn   : render function (render_with_dynamic_gaussians_mask)
    pipe        : PipelineParams
    background  : torch.Tensor [3] background colour
    cam_type    : scene.dataset_type string
    model_path  : experiment output directory (used for the output sub-folder)
    threshold_sigma : frames with loss > mean + k*std are flagged (default 1.0)

    Returns
    -------
    set[int]  -- camera UIDs to be doubled in the fine_coloring pool.
    """
    print("[importance_sampling] Computing per-frame RGB L1 loss on all training views ...")
    train_cams = scene.getTrainCameras()

    losses       = []
    cam_ids      = []
    rendered_imgs = []
    gt_imgs      = []

    with torch.no_grad():
        for cam in train_cams:
            cam.to_device("cuda")
            pkg      = render_fn(cam, gaussians, pipe, background,
                                 stage="fine_coloring", cam_type=cam_type,
                                 training=False)
            rendered = pkg["render"]                        # [3, H, W]
            gt       = cam.original_image.cuda()            # [3, H, W]
            loss_val = torch.mean(torch.abs(rendered - gt[:3])).item()

            losses.append(loss_val)
            cam_ids.append(cam.uid)
            rendered_imgs.append(rendered.cpu())
            gt_imgs.append(gt.cpu())

    losses_t   = torch.tensor(losses)
    mean_loss  = losses_t.mean().item()
    std_loss   = losses_t.std().item()
    threshold  = mean_loss + threshold_sigma * std_loss

    # ------------------------------------------------------------------
    # Build sorted high-loss list and persist visualisations
    # ------------------------------------------------------------------
    high_loss_data = [
        (uid, lv, rend, gt_)
        for uid, lv, rend, gt_ in zip(cam_ids, losses, rendered_imgs, gt_imgs)
        if lv > threshold
    ]
    high_loss_data.sort(key=lambda x: x[1], reverse=True)

    output_dir = os.path.join(model_path, "high_loss_training_views")
    os.makedirs(output_dir, exist_ok=True)

    for rank, (uid, lv, rendered, gt_) in enumerate(high_loss_data, 1):
        rendered_np = (torch.clamp(rendered[:3], 0, 1)
                       .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_np       = (torch.clamp(gt_[:3], 0, 1)
                       .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        comparison  = np.hstack([rendered_np, gt_np])
        filename    = f"{rank:03d}_uid{uid:05d}_loss{lv:.4f}.png"
        cv2.imwrite(os.path.join(output_dir, filename),
                    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    high_loss_ids = {uid for uid, _, _, _ in high_loss_data}

    print(f"[importance_sampling] mean={mean_loss:.4f}  std={std_loss:.4f}  "
          f"threshold={threshold:.4f}  "
          f"high-loss frames: {len(high_loss_ids)}/{len(train_cams)} "
          f"({100 * len(high_loss_ids) / max(len(train_cams), 1):.1f}%)")
    print(f"[importance_sampling] Visualisations saved → {output_dir}")

    return high_loss_ids


def build_importance_sampled_pool(base_pool, high_loss_cam_ids):
    """Return a new sampling pool where high-loss cameras appear twice.

    Parameters
    ----------
    base_pool        : list[Camera]  – deep-copy of the full training cam list
                       (i.e. ``temp_list`` in train_dynamic_depth.py)
    high_loss_cam_ids : set[int]     – UIDs returned by compute_high_loss_cam_ids

    Returns
    -------
    list[Camera]  -- extended pool; high-loss cams appear once extra at the end.
    """
    if not high_loss_cam_ids:
        return base_pool

    extras = [copy.deepcopy(cam) for cam in base_pool
              if cam.uid in high_loss_cam_ids]
    pool   = base_pool + extras

    print(f"[importance_sampling] Duplicated {len(extras)} high-loss cameras "
          f"in fine_coloring pool  (pool size: {len(pool)}, "
          f"of which {len(extras)} are duplicates).")
    return pool
