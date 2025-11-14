#!/bin/bash
# Debug script: Full loss reinforcement with stronger supervision
# Purpose: Test with reinforced losses (depth, RGB, normal) - slightly longer training
# Focus: Reinforce entire normal map, depth map, RGB map, and everything uniformly

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

# ====================================================================
# TRAINING COMMAND
# ====================================================================
python train_dynamic_depth.py \
  --batch_size 4 \
  --background_depth_iterations 1 \
  --background_RGB_iterations 500 \
  --dynamics_depth_iterations 500 \
  --dynamics_RGB_iterations 500 \
  --fine_iterations 500 \
  --normal_loss_weight 0.5 \
  --general_depth_weight 10.0 \
  --rgb_weight 30.0 \
  --scale_loss_weight 100.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 100.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/ADT/default.py" \
  --port 6254 \
  --expname recognition_inital_debug \
  --source_path "data/ADT/recognition/colmap" \
