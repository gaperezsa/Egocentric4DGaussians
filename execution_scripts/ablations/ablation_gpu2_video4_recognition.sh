#!/bin/bash
# Ablation script for GPU 2: HOI4D Video4 + ADT recognition
# Two ablations per sequence:
#   A) No chamfer/dynamics-depth stage (dynamics_depth_iterations=1, skip straight to dynamics RGB)
#   B) No depth supervision in loss (general_depth_weight=0, normal_loss_weight=0)
# All experiments start from scratch (no --start_checkpoint).

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

# ====================================================================
# [1/4] HOI4D Video4 - Ablation: no chamfer stage (dynamics_depth_iterations=1)
# ====================================================================
echo "=========================================================="
echo "[1/4] HOI4D Video4 - no chamfer stage"
echo "=========================================================="
python train_dynamic_depth.py \
  --batch_size 4 \
  --background_depth_iterations 10400 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 1 \
  --dynamics_RGB_iterations 4100 \
  --fine_iterations 8300 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 0.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/HOI4D/video4_all_dyn_fine.py" \
  --port 6278 \
  --expname ablation_video4_no_chamfer_stage \
  --source_path "data/HOI4D/Video4/colmap" \
  --start_checkpoint "output/ablation_video4_no_chamfer_stage/chkpnt_background_depth_10400.pth" \
  --all_dynamic_on_fine

# ====================================================================
# [2/4] ADT recognition - Ablation: no chamfer stage (dynamics_depth_iterations=1)
# ====================================================================
echo "=========================================================="
echo "[2/4] ADT recognition - no chamfer stage"
echo "=========================================================="
python train_dynamic_depth.py \
  --batch_size 4 \
  --background_depth_iterations 20800 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 1 \
  --dynamics_RGB_iterations 4100 \
  --fine_iterations 10400 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 0.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/ADT/fine_all_dyn.py" \
  --port 6279 \
  --expname ablation_recognition_no_chamfer_stage \
  --source_path "data/ADT/recognitionV2/colmap"

# ====================================================================
# [3/4] HOI4D Video4 - Ablation: no depth/normal loss (weights set to 0)
# ====================================================================
echo "=========================================================="
echo "[3/4] HOI4D Video4 - no depth/normal loss"
echo "=========================================================="
python train_dynamic_depth.py \
  --batch_size 4 \
  --background_depth_iterations 10400 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 4100 \
  --dynamics_RGB_iterations 4100 \
  --fine_iterations 8300 \
  --normal_loss_weight 0.0 \
  --general_depth_weight 0.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/HOI4D/video4_all_dyn_fine.py" \
  --port 6280 \
  --expname ablation_video4_no_depth_loss \
  --source_path "data/HOI4D/Video4/colmap" \
  --all_dynamic_on_fine

# ====================================================================
# [4/4] ADT recognition - Ablation: no depth/normal loss (weights set to 0)
# ====================================================================
echo "=========================================================="
echo "[4/4] ADT recognition - no depth/normal loss"
echo "=========================================================="
python train_dynamic_depth.py \
  --batch_size 4 \
  --background_depth_iterations 20800 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 8300 \
  --dynamics_RGB_iterations 4100 \
  --fine_iterations 10400 \
  --normal_loss_weight 0.0 \
  --general_depth_weight 0.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --configs "arguments/ADT/fine_all_dyn.py" \
  --port 6281 \
  --expname ablation_recognition_no_depth_loss \
  --source_path "data/ADT/recognitionV2/colmap"
