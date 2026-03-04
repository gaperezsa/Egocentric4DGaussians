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
  --background_depth_iterations 20800 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 8300 \
  --dynamics_RGB_iterations 4100 \
  --fine_iterations 10400 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --all_dynamic_on_fine \
  --black_background \
  --configs "arguments/ADT/fine_all_dyn_importance_sampling.py" \
  --port 6260 \
  --expname clean_all_dyn \
  --source_path "data/ADT/cleanV2/colmap"