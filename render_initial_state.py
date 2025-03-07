import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, render_with_dynamic_gaussians_mask,  network_gui
import sys
from scene import Scene, GaussianModel
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

import wandb

def prepare_output(expname):    
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

def scene_reconstruction(dataset, opt, hyper, pipe, checkpoint, gaussians, scene, stage, timer):

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    viewpoint_stack = None
    # lpips_model = lpips.LPIPS(net="alex").cuda()


    #if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        #viewpoint_stack = [i for i in train_cams]
        #temp_list = copy.deepcopy(viewpoint_stack)
    # 
    batch_size = opt.batch_size
    
    viewpoint_stack = scene.getTrainCameras()
    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,num_workers=0,collate_fn=list)
    print("data loading done")
    for i, viewpoint_cam in enumerate(viewpoint_stack_loader):    

        render_training_image(scene, gaussians, viewpoint_cam, render_with_dynamic_gaussians_mask, pipe, background, stage, i,timer.get_elapsed_time(),scene.dataset_type)
        
def render_initial_state(dataset, hyper, opt, pipe, expname, args):
    # first_iter = 0
    prepare_output(expname)
    if args.load_iteration or args.start_checkpoint:
        stage="test_close_final"
    else:
        stage="initial_state_coarse"
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration, load_coarse=None, init_random_pcd = args.init_random_pcd)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, args.start_checkpoint, gaussians, scene, stage, timer)



if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")

    parser.add_argument("--netork_width", type=int, default = 128)
    parser.add_argument("--bs", type=int, default = 16)
    parser.add_argument('--init_random_pcd', action='store_true', default=False)

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("opening " + args.model_path)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Manual setting for sweeps
    op_params = op.extract(args)
    op_params.batch_size =  args.bs

    hp_params = hp.extract(args)
    hp_params.net_width =  args.netork_width

    render_initial_state(lp.extract(args), hp_params, op_params, pp.extract(args), args.expname,args)

    # All done
    print("\Rendering complete.")