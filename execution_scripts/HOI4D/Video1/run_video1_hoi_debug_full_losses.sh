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
  --background_depth_iterations 500 \
  --background_RGB_iterations 500 \
  --dynamics_depth_iterations 1000 \
  --dynamics_RGB_iterations 2000 \
  --fine_iterations 3000 \
  --normal_loss_weight 0.5 \
  --general_depth_weight 10.0 \
  --rgb_weight 30.0 \
  --scale_loss_weight 100.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 100.0 \
  --plane_tv_weight 0.001 \
  --configs "/home/gperezsantamaria/gperezsantamaria2/Egocentric4DGaussians/arguments/HOI4D/default.py" \
  --port 6254 \
  --expname video1_normals_with_gsplat_debug \
  --source_path "/home/gperezsantamaria/gperezsantamaria2/Egocentric4DGaussians/data/automatic_data_extraction_testing/with_monst3r/Video1/colmap" \
