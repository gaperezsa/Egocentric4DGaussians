#!/usr/bin/env python3
"""Render-only timed FPS + peak render VRAM for Egocentric4DGaussians (OUR method).

Loads the FINAL fine_coloring checkpoint (which preserves _dynamic_xyz + the
deformation network) rather than the point_cloud PLY (load_ply zeroes the dynamic
mask and the max-iteration PLY is the background stage, not the final model).

Times ONLY the rasterization call render_with_dynamic_gaussians_mask(view, ...)
over the test cameras, with torch.cuda.synchronize() around the timed region and a
few warmup frames excluded. Prints machine-parseable OURS_* lines.
"""
import os, sys, glob, re, argparse
from time import time
import torch

sys.argv_backup = sys.argv
p = argparse.ArgumentParser()
p.add_argument("--exp", required=True)
p.add_argument("--warmup", type=int, default=3)
cli, _ = p.parse_known_args()
EXP = cli.exp
MODEL_PATH = f"./output/{EXP}"

# ---- reconstruct full args from cfg_args + config file (like render.py) ----
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from utils.general_utils import safe_state

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
hyperparam = ModelHiddenParams(parser)
parser.add_argument("--exp")
parser.add_argument("--warmup", type=int, default=3)
# inject model_path so get_combined_args finds cfg_args
sys.argv = ["render_fps_ours.py", "--model_path", MODEL_PATH]
args = get_combined_args(parser)
if getattr(args, "configs", None):
    import mmengine
    from utils.params_utils import merge_hparams
    config = mmengine.Config.fromfile(args.configs)
    args = merge_hparams(args, config)
safe_state(True)

def emit(msg):
    sys.stderr.write(msg + "\n"); sys.stderr.flush()

from scene import Scene
from gaussian_renderer import GaussianModel, render_with_dynamic_gaussians_mask

dataset = model.extract(args)
hyp = hyperparam.extract(args)
pipe = pipeline.extract(args)

# pick final fine_coloring checkpoint (max stage-iter)
it = lambda q: int(q.split("_")[-1].split(".")[0])
cks = glob.glob(f"{MODEL_PATH}/chkpnt_fine_coloring_*.pth")
assert cks, f"no fine_coloring checkpoint in {MODEL_PATH}"
ckpt = max(cks, key=it)
fine_iter = it(ckpt)

gaussians = GaussianModel(dataset.sh_degree, hyp)
# Scene loads cameras (and a ply model we immediately overwrite via restore).
# Use load_iteration=fine_iter so it reads the fine-stage ply dir (cameras only matter).
scene = Scene(dataset, gaussians, load_iteration=fine_iter, shuffle=False)
cam_type = scene.dataset_type

# authoritative final model from checkpoint (restores _xyz, _dynamic_xyz, deformation)
model_args, _ = torch.load(ckpt, map_location="cpu")
gaussians.restore(model_args, None, "fine")
# move to cuda
for a in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation", "_dynamic_xyz"]:
    t = getattr(gaussians, a)
    setattr(gaussians, a, t.cuda())
gaussians._deformation = gaussians._deformation.cuda()
if hasattr(gaussians, "_deformation_table"):
    gaussians._deformation_table = gaussians._deformation_table.cuda()

N = gaussians._xyz.shape[0]
ndyn = int(gaussians._dynamic_xyz.sum().item())
nstat = N - ndyn

bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg, dtype=torch.float32, device="cuda")

test_cams = scene.getTestCameras()
n_test = len(test_cams)
emit(f"OURS_INFO exp={EXP} ckpt={os.path.basename(ckpt)} N={N} dyn={ndyn} stat={nstat} n_test={n_test}")

if n_test == 0:
    emit("OURS_RESULT exp=%s FPS=NA reason=no_test_cameras" % EXP)
    sys.exit(0)

W = H = None
with torch.no_grad():
    # warmup
    for i in range(min(cli.warmup, n_test)):
        out = render_with_dynamic_gaussians_mask(test_cams[i], gaussians, pipe, background, cam_type=cam_type)
        if W is None:
            H, W = out["render"].shape[-2], out["render"].shape[-1]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time()
    for view in test_cams:
        out = render_with_dynamic_gaussians_mask(view, gaussians, pipe, background, cam_type=cam_type)
    torch.cuda.synchronize()
    t1 = time()

secs = t1 - t0
fps = n_test / secs
peak_render_mb = torch.cuda.max_memory_allocated() / 1e6
emit(f"OURS_RESULT exp={EXP} FPS={fps:.4f} render_seconds={secs:.4f} n_frames={n_test} "
      f"resolution={W}x{H} peak_vram_render_mb={peak_render_mb:.1f} N={N} dyn={ndyn} stat={nstat}")
