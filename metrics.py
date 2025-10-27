from pathlib import Path
import os, json
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim
import numpy as np
import matplotlib as mpl
import imageio.v2 as imageio

# ------------- helpers -------------
def _open_and_crop_rgb(img_path: Path, crop: int) -> torch.Tensor:
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if crop > 0:
        w, h = img.size
        eff = min(crop, max((w-1)//2, 0), max((h-1)//2, 0))
        if eff > 0:
            img = img.crop((eff, eff, w-eff, h-eff))
    t = tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].cuda()
    return t

def _crop_tensor_2d(arr: torch.Tensor, crop: int) -> torch.Tensor:
    if crop <= 0:
        return arr
    H, W = arr.shape[-2], arr.shape[-1]
    eff = min(crop, max((W-1)//2, 0), max((H-1)//2, 0))
    if eff <= 0:
        return arr
    return arr[..., eff:H-eff, eff:W-eff]

def _save_fixed_jet_np(depth_m_np: np.ndarray, out_png: str, vmin: float = 0.0, vmax: float = 2.5):
    cmap = mpl.cm.get_cmap('jet')
    d = depth_m_np.copy()
    nonpos = d <= 0.0
    d = np.clip(d, vmin, vmax)
    denom = max(vmax - vmin, 1e-6)
    norm = (d - vmin) / denom
    vis = (cmap(norm)[..., :3] * 255.0).astype(np.uint8)
    vis[nonpos] = 0
    imageio.imwrite(out_png, vis)

def _generate_fixed_depth_visuals(folder: Path, names: list, vmin=0.0, vmax=2.5):
    """
    If fixed-range visuals are missing, create them from tensors:
      - depth_renders_tensors/*.pt  -> depth_metric_vis_fixed/*.png (+ alias aligned_metric_vis_fixed)
      - depth_gt_tensors/*.pt       -> depth_gt_vis_fixed/*.png
    """
    pred_pt_dir = folder / "depth_renders_tensors"
    gt_pt_dir   = folder / "depth_gt_tensors"
    pred_vis_dir = folder / "depth_metric_vis_fixed"
    alias_vis_dir = folder / "aligned_metric_vis_fixed"
    gt_vis_dir   = folder / "depth_gt_vis_fixed"
    pred_vis_dir.mkdir(parents=True, exist_ok=True)
    alias_vis_dir.mkdir(parents=True, exist_ok=True)
    gt_vis_dir.mkdir(parents=True, exist_ok=True)

    for n in names:
        p = pred_pt_dir / f"{n}.pt"
        if p.is_file():
            d = torch.load(p)
            while d.dim() > 2: d = d.squeeze(0)
            dnp = d.detach().cpu().numpy().astype(np.float32)
            _save_fixed_jet_np(dnp, str(pred_vis_dir / f"{n}.png"), vmin, vmax)
            _save_fixed_jet_np(dnp, str(alias_vis_dir / f"{n}.png"), vmin, vmax)

        g = gt_pt_dir / f"{n}.pt"
        if g.is_file():
            t = torch.load(g)
            while t.dim() > 2: t = t.squeeze(0)
            tnp = t.detach().cpu().numpy().astype(np.float32)
            _save_fixed_jet_np(tnp, str(gt_vis_dir / f"{n}.png"), vmin, vmax)

# ------------- original API (kept) -------------
def readImages(renders_dir, gt_dir):
    renders, gts, image_names = [], [], []
    for fname in os.listdir(renders_dir):
        render = Image.open(Path(renders_dir) / fname)
        gt     = Image.open(Path(gt_dir) / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    print("")
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
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims, psnrs, lpipss, lpipsa, ms_ssims, Dssims = [], [], [], [], [], []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                    ms_ssims.append(ms_ssim(renders[idx], gts[idx], data_range=1, size_average=True))
                    lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                    Dssims.append((1 - ms_ssims[-1]) / 2)

                print("Scene:", scene_dir, "SSIM: {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene:", scene_dir, "PSNR: {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("Scene:", scene_dir, "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("Scene:", scene_dir, "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                print("Scene:", scene_dir, "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                print("Scene:", scene_dir, "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                    "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                    "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                    "D-SSIM": torch.tensor(Dssims).mean().item()
                })
                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: v for v, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: v for v, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS-vgg": {name: v for v, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "LPIPS-alex": {name: v for v, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                    "MS-SSIM": {name: v for v, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                    "D-SSIM": {name: v for v, name in zip(torch.tensor(Dssims).tolist(), image_names)}
                })

            with open(os.path.join(scene_dir, "results.json"), 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(os.path.join(scene_dir, "per_view.json"), 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            raise e

# ------------- unified folder evaluator (use this) -------------
def evaluate_single_folder(folder_path: str, crop_px: int = 5, suffix: str = "cropped5px"):
    """
    Evaluate a single folder that contains:
      folder/
        ├─ renders/                (PNG; names must match GT)
        ├─ gt/                     (PNG)
        ├─ depth_renders_tensors/  (pred depth .pt)
        └─ depth_gt_tensors/       (true GT .pt)  <-- auto-materialized if missing

    If depth_gt_tensors/ is missing, we read render_manifest.json and try to
    create it from raw sparse GT (sibling 'sparse_unprocessed_gt_depth').
    """
    from utils.depth_gt_utils import ensure_true_depth_gt  # lazy import

    folder = Path(folder_path)
    renders_dir = folder / "renders"
    gt_dir = folder / "gt"
    if not renders_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Expected 'renders' and 'gt' under {folder}")

    # Try to materialize true depth if it's not there yet
    depth_gt_tensors = folder / "depth_gt_tensors"
    if not depth_gt_tensors.exists() or not any(depth_gt_tensors.iterdir()):
        manifest_path = folder / "render_manifest.json"
        if manifest_path.is_file():
            try:
                man = json.load(open(manifest_path, "r"))
                src = man.get("source_path", None)
                names = man.get("image_names", [])
                W = int(man.get("width", 0)); H = int(man.get("height", 0))
                if src and names and W > 0 and H > 0:
                    print("[metrics] depth_gt_tensors missing — materializing from true sparse GT…")
                    ensure_true_depth_gt(str(folder), src, names, (W, H))
            except Exception as e:
                print(f"[metrics] Failed to materialize true depth GT: {e}")

    # ---------- ensure fixed-range jet depth visuals (0–2.5 m) ----------
    # build the name list from renders (they define the stems)
    image_names = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(".png")])
    stems = [os.path.splitext(n)[0] for n in image_names]
    try:
        _generate_fixed_depth_visuals(folder, stems, vmin=0.0, vmax=2.5)
    except Exception as e:
        print(f"[metrics] fixed-range depth vis skipped: {e}")

    # ---------- RGB ----------
    image_names = sorted([f for f in os.listdir(renders_dir) if f.lower().endswith(".png")])
    renders, gts = [], []
    for fname in image_names:
        r_t = _open_and_crop_rgb(renders_dir / fname, crop_px)
        g_t = _open_and_crop_rgb(gt_dir / fname, crop_px)
        renders.append(r_t); gts.append(g_t)

    ssims, psnrs, lpipss, lpipsa, ms_ssims, Dssims = [], [], [], [], [], []
    for i in tqdm(range(len(renders)), desc=f"RGB metrics (crop {crop_px}px)"):
        r, g = renders[i], gts[i]
        ssims.append(ssim(r, g))
        psnrs.append(psnr(r, g))
        lpipss.append(lpips(r, g, net_type='vgg'))
        ms_ssims.append(ms_ssim(r, g, data_range=1, size_average=True))
        lpipsa.append(lpips(r, g, net_type='alex'))
        Dssims.append((1 - ms_ssims[-1]) / 2)

    print("SSIM (crop): {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("PSNR (crop): {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("LPIPS-vgg (crop): {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("LPIPS-alex (crop): {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
    print("MS-SSIM (crop): {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
    print("D-SSIM (crop): {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

    full_dict = {
        "SSIM": torch.tensor(ssims).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
        "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
        "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
        "D-SSIM": torch.tensor(Dssims).mean().item(),
        "crop_px": crop_px
    }
    per_view_dict = {
        "SSIM":       {name: v for v, name in zip(torch.tensor(ssims).tolist(), image_names)},
        "PSNR":       {name: v for v, name in zip(torch.tensor(psnrs).tolist(), image_names)},
        "LPIPS-vgg":  {name: v for v, name in zip(torch.tensor(lpipss).tolist(), image_names)},
        "LPIPS-alex": {name: v for v, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
        "MS-SSIM":    {name: v for v, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
        "D-SSIM":     {name: v for v, name in zip(torch.tensor(Dssims).tolist(), image_names)},
        "crop_px": crop_px
    }

    # ---------- Depth (true GT) ----------
    depth_gt_dir = folder / "depth_gt_tensors"
    depth_pred_dir = folder / "depth_renders_tensors"
    if depth_gt_dir.exists() and depth_pred_dir.exists():
        mse_vals, rmse_vals = [], []
        mse_per, rmse_per = {}, {}
        depth_names = sorted([f for f in os.listdir(depth_pred_dir) if f.lower().endswith(".pt")])
        for fname in tqdm(depth_names, desc=f"Depth metrics (crop {crop_px}px)"):
            pr_t = torch.load(depth_pred_dir / fname).to(torch.float32)
            gt_t = torch.load(depth_gt_dir / fname).to(torch.float32)
            while pr_t.dim() > 2: pr_t = pr_t.squeeze(0)
            while gt_t.dim() > 2: gt_t = gt_t.squeeze(0)
            H = min(pr_t.shape[0], gt_t.shape[0]); W = min(pr_t.shape[1], gt_t.shape[1])
            pr_t, gt_t = pr_t[:H,:W], gt_t[:H,:W]
            pr_t = _crop_tensor_2d(pr_t, crop_px); gt_t = _crop_tensor_2d(gt_t, crop_px)
            valid = (gt_t > 0).to(torch.float32); n = valid.sum()
            if n <= 0: continue
            err = (pr_t - gt_t)
            mse = ((err**2) * valid).sum() / n
            rmse = torch.sqrt(torch.clamp(mse, min=0.0))
            mse_v, rmse_v = float(mse.item()), float(rmse.item())
            mse_vals.append(mse_v); rmse_vals.append(rmse_v)
            mse_per[fname] = mse_v; rmse_per[fname] = rmse_v

        if mse_vals:
            import numpy as _np
            full_dict.update({
                "Depth MSE": float(_np.mean(mse_vals)),
                "Depth RMSE": float(_np.mean(rmse_vals))
            })
            per_view_dict.update({
                "Depth MSE": mse_per,
                "Depth RMSE": rmse_per
            })
    else:
        print("[metrics] depth_gt_tensors or depth_renders_tensors missing; depth metrics skipped.")

    out_res = folder / f"results_{suffix}.json"
    out_per = folder / f"per_view_{suffix}.json"
    with open(out_res, "w") as fp: json.dump(full_dict, fp, indent=True)
    with open(out_per, "w") as fp: json.dump(per_view_dict, fp, indent=True)

# ------------- optional convenience over a model root -------------
def evaluate_model_root_for_splits(root_dir: str, crop_px: int = 5):
    """
    Walks a model output root and evaluates known split subfolders if present:
      <root>/final_split_renders/split_{train,test,eval_static,eval_dynamic}/ours_*/{renders,gt,...}
      <root>/final_sequence_render/sequence_full/ours_*/{renders,gt,...}
    """
    root = Path(root_dir)
    split_base = root / "final_split_renders"
    full_base  = root / "final_sequence_render" / "sequence_full"
    todo = []

    if split_base.exists():
        for split in ["split_train", "split_test", "split_eval_static", "split_eval_dynamic"]:
            sdir = split_base / split
            if sdir.exists():
                for ours in sorted(sdir.glob("ours_*")):
                    if (ours / "renders").exists():
                        todo.append(str(ours))

    if full_base.exists():
        for ours in sorted(full_base.glob("ours_*")):
            if (ours / "renders").exists():
                todo.append(str(ours))

    for folder in todo:
        try:
            evaluate_single_folder(folder, crop_px=crop_px)
        except Exception as e:
            print(f"[WARN] metrics failed for {folder}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    parser = ArgumentParser(description="Metrics")
    parser.add_argument('--model_paths', '-m', required=False, nargs="+", type=str, default=[])
    parser.add_argument('--folder', type=str, default="")
    parser.add_argument('--crop_px', type=int, default=5)
    args = parser.parse_args()

    if args.folder:
        evaluate_single_folder(args.folder, crop_px=args.crop_px)
    elif args.model_paths:
        evaluate(args.model_paths)
    else:
        print("Provide --folder <path> (preferred) or --model_paths ... (legacy).")