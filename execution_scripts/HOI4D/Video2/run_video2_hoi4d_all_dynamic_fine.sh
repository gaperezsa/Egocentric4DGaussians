#!/bin/bash
# HOI4D Video3 - Ablation: all_dynamic_on_fine
# Purpose: Same as DA3 baseline but with --all_dynamic_on_fine enabled.
#   At the fine_coloring stage ALL Gaussians (static + dynamic) are marked as
#   dynamic so the deformation network covers the full scene.  Static Gaussians
#   keep their colour/position but receive the lower dynamic learning rate,
#   allowing them to learn small corrections while avoiding large drift.
# Comparison target: run_video3_hoi4d_da3.sh  (identical hyper-params, only flag differs)

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

# ====================================================================
# TRAINING COMMAND
# ====================================================================
python train_dynamic_depth.py \
  --batch_size 4 \
  --background_depth_iterations 10400 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 4100 \
  --dynamics_RGB_iterations 4100 \
  --fine_iterations 10400 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/HOI4D/video2_all_dyn_fine.py" \
  --port 6257 \
  --expname video2_all_dynamic_fine_importance_sampling_stop_pruning \
  --source_path "data/HOI4D/Video2/colmap" \
  --start_checkpoint "/home/gperezsantamaria/sda_data/ICML_submission/Egocentric4DGaussians/output/video2_time_smoothed/chkpnt_dynamics_RGB_4100.pth" \
  --all_dynamic_on_fine \
  --black_background \
