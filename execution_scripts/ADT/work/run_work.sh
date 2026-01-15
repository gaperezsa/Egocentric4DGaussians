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
  --batch_size 16 \
  --background_depth_iterations 4000 \
  --background_RGB_iterations 4000 \
  --dynamics_depth_iterations 2000 \
  --dynamics_RGB_iterations 4000 \
  --fine_iterations 4000 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/ADT/default.py" \
  --port 6263 \
  --expname work_v2 \
  --source_path "data/ADT/workV2/colmap" \
