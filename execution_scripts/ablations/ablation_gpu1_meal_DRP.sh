#!/bin/bash
# Ablation script for GPU 1: ADT meal  --  DRP (Depth Blame Pruning removed)
#   No Depth Blame Pruning: depth_blame_percent=0.0 (our special pruner completely disabled)
# All other hyper-parameters match the best/full training configuration.
# All experiments start from scratch (no --start_checkpoint).

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

# ====================================================================
# ADT meal - Ablation: no depth blame pruning (DRP)
# ====================================================================
echo "=========================================================="
echo "ADT meal - no depth blame pruning (DRP)"
echo "=========================================================="
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
  --depth_blame_percent 0.0 \
  --all_dynamic_on_fine \
  --black_background \
  --configs "arguments/ADT/fine_all_dyn_importance_sampling.py" \
  --port 6293 \
  --expname ablation_true_meal_no_depth_blame_pruning \
  --source_path "data/ADT/mealV2/colmap"
