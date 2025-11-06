#!/bin/bash
# Get the repository root directory

# Activate conda environment (optional, will warn if already active)
eval "$(conda shell.bash hook)"
conda activate Gaussians4D

python train_dynamic_depth.py --background_depth_iter 2000 --background_RGB_iter 5000 --bs 8 --chamfer_weight 145 --configs "/home/gperezsantamaria/gperezsantamaria2/Egocentric4DGaussians/arguments/HOI4D/default.py" --densification_interval 200  --dynamics_depth_iter 8000 --dynamics_RGB_iter 2000 --fine_iter 5000 --fine_opt_dyn_lr_downscaler 0.4350902858537733 --port 6258 --pruning_interval 670 --source_path "/home/gperezsantamaria/gperezsantamaria2/Egocentric4DGaussians/data/HOI4D/officialHOI4DVideo1/colmap"  --expname video1_bcv009_fulltrain --start_checkpoint "/home/gperezsantamaria/gperezsantamaria2/Egocentric4DGaussians/output/video1_bcv009_fulltrain/chkpnt_dynamics_RGB_2000.pth" 