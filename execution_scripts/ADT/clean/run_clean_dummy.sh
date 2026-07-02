#!/bin/bash
# Dummy smoke-test script: same params as run_clean.sh but with all stage
# iteration counts brought down to trivial numbers, just to verify the
# current codebase + freshly rebuilt submodules run end to end.

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ====================================================================
# TRAINING COMMAND (dummy iteration counts)
# ====================================================================
python train_dynamic_depth.py \
  --batch_size 2 \
  --background_depth_iterations 500 \
  --background_RGB_iterations 0 \
  --dynamics_depth_iterations 500 \
  --dynamics_RGB_iterations 500 \
  --fine_iterations 500 \
  --normal_loss_weight 0.4 \
  --general_depth_weight 10.0 \
  --rgb_weight 80.0 \
  --scale_loss_weight 300.0 \
  --ssim_weight 0.5 \
  --chamfer_weight 1.0 \
  --plane_tv_weight 0.001 \
  --test_iterations 100 300 \
  --checkpoint_iterations 100 300 \
  --all_dynamic_on_fine \
  --black_background \
  --configs "arguments/ADT/fine_all_dyn_importance_sampling.py" \
  --port 6261 \
  --expname clean_all_dyn_dummy \
  --source_path "data/ADT/cleanV2/colmap"
