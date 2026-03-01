#!/bin/bash
# HOI4D Video5 - Time smoothness debug run
# background_RGB stage skipped (0 iter)
# time_smoothness_weight and l1_time_planes come from arguments/HOI4D/default.py

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

cd /home/gperezsantamaria/sda_data/ICML_submission/Egocentric4DGaussians

python train_dynamic_depth.py \
  --batch_size 16 \
  --background_depth_iterations 10400 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 4100 \
  --dynamics_RGB_iterations 4100 \
  --fine_iterations 6200 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/HOI4D/default.py" \
  --port 6285 \
  --expname video5_time_smoothed_more_dyn \
  --source_path "data/HOI4D/Video5/colmap" \
  --start_checkpoint "/home/gperezsantamaria/sda_data/ICML_submission/Egocentric4DGaussians/output/video5_time_smoothed/chkpnt_background_depth_10400.pth"
