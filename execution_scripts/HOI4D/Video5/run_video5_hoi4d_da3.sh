#!/bin/bash
# HOI4D Video5 - Training with DA3 depth predictions
# Purpose: Train EgoGaussian with DA3 metric depth (DA3 wins on this video)
# Features: Full loss supervision with depth, RGB, and normal guidance

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

# ====================================================================
# TRAINING COMMAND
# ====================================================================
python train_dynamic_depth.py \
  --batch_size 16 \
  --background_depth_iterations 4100 \
  --background_RGB_iterations 6200 \
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
  --port 6260 \
  --expname video5_EgoGaussian_background \
  --source_path "data/HOI4D/Video5/colmap" \
