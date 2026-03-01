#!/bin/bash
# HOI4D Video1 - Training with DA3 depth predictions
# Purpose: Train EgoGaussian with DA3 metric depth (superior to monst3r on Video3)
# Features: Full loss supervision with depth, RGB, and normal guidance

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

# ====================================================================
# TRAINING COMMAND
# ====================================================================
python train_dynamic_depth.py \
  --batch_size 16 \
  --background_depth_iterations 10400 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 4100 \
  --dynamics_RGB_iterations 4100 \
  --fine_iterations 4100 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/HOI4D/default.py" \
  --port 6217 \
  --expname video1_dyn_debug \
  --source_path "data/HOI4D/Video1/colmap" \
  --start_checkpoint "/home/gperezsantamaria/sda_data/ICML_submission/Egocentric4DGaussians/output/video1_EgoGaussian_background_fix/chkpnt_dynamics_RGB_4100.pth"
