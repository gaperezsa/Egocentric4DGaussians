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
  --background_depth_iterations 2000 \
  --background_RGB_iterations 3000 \
  --dynamics_depth_iterations 2000 \
  --dynamics_RGB_iterations 2000 \
  --fine_iterations 3000 \
  --normal_loss_weight 0.5 \
  --general_depth_weight 10.0 \
  --rgb_weight 30.0 \
  --scale_loss_weight 100.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 100.0 \
  --plane_tv_weight 0.001 \
  --start_checkpoint /home/gperezsantamaria/sda_data/Egocentric4DGaussians/output/recognition_initial_debug4/chkpnt_background_RGB_1999.pth \
  --configs "arguments/ADT/default.py" \
  --port 6258 \
  --expname recognition_initial_debug4 \
  --source_path "data/ADT/recognition/colmap" \
