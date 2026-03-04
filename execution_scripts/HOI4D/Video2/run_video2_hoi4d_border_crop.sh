#!/bin/bash
# HOI4D Video2 - Ablation: border_crop_fine
# Purpose: Same as video2_all_dynamic_fine_importance_sampling but with 4px left
#   and 1px bottom border pixels excluded from ALL fine_coloring losses
#   (RGB L1, depth, SSIM).  Border pixels were identified as near-black sensor
#   artifacts present in every GT frame of Video2 (4px left, 1px bottom).
#
# Implementation: spatial boolean mask built from border_crop_left=4 /
#   border_crop_bottom=1 params.  Tensor shapes are UNCHANGED — the mask is
#   applied only inside loss functions, so no code outside the loss block
#   needs to change.  gradient_aware_depth_loss already accepts mask=;
#   RGB/depth plain L1 uses l1_filtered_loss; SSIM uses new masked_ssim().
#
# Comparison target: run_video2_hoi4d_all_dynamic_fine.sh
#   (identical hyper-params and checkpoint, only border masking differs)

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

# ====================================================================
# TRAINING COMMAND
# ====================================================================
python train_dynamic_depth.py \
  --batch_size 4 \
  fine_opt_dyn_lr_downscaler = 1
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/HOI4D/video2_border_crop.py" \
  --port 6258 \
  --expname video2_border_crop_fine \
  --source_path "data/HOI4D/Video2/colmap" \
  --start_checkpoint "/home/gperezsantamaria/sda_data/ICML_submission/Egocentric4DGaussians/output/video2_time_smoothed/chkpnt_dynamics_RGB_4100.pth" \
  --all_dynamic_on_fine \
  --black_background \
