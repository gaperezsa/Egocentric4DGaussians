#!/bin/bash
# Debug script: Resume from background_RGB checkpoint → start at dynamics_depth stage
# Purpose: Debug dynamic Gaussian spawn quality (shapes, sizes, positions)

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

# ====================================================================
# TRAINING COMMAND
# ====================================================================
python train_dynamic_depth.py \
  --batch_size 4 \
  --background_depth_iterations 1000 \
  --background_RGB_iterations 1000 \
  --dynamics_depth_iterations 1000 \
  --dynamics_RGB_iterations 1000 \
  --fine_iterations 1000 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --all_dynamic_on_fine \
  --black_background \
  --configs "arguments/ADT/default.py" \
  --port 6264 \
  --expname clean_all_dyn_debug \
  --source_path "data/ADT/cleanV2/colmap" \
  --start_checkpoint /home/gperezsantamaria/sda_data/ICML_submission/Egocentric4DGaussians/output/clean_all_dyn_debug/chkpnt_background_RGB_1000.pth
