#!/bin/bash
# Test script: Config provides defaults, CLI overrides selected params

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

echo "======================================================================"
echo "PARAMETER FLOW TEST"
echo "======================================================================"
echo ""
echo "Config file provides (from HOI4D/default.py):"
echo "  - batch_size: 2"
echo "  - background_depth_iterations: 500"
echo "  - background_RGB_iterations: 300"
echo "  - dynamics_depth_iterations: 200"
echo "  - dynamics_RGB_iterations: 200"
echo "  - fine_iterations: 300"
echo "  - pruning_interval: 700"
echo "  - densification_interval: 400"
echo "  - chamfer_weight: 50.0"
echo "  - general_depth_weight: 0.01"
echo "  - plane_tv_weight: 0.0001"
echo ""
echo "CLI (this bash script) OVERRIDES only these:"
echo "  - batch_size: 4 (was 2 in config)"
echo "  - pruning_interval: 100 (was 700 in config)"
echo "  - chamfer_weight: 75.0 (was 50.0 in config)"
echo ""
echo "Expected result after merge:"
echo "  - batch_size: 4 (CLI override)"
echo "  - pruning_interval: 100 (CLI override)"
echo "  - chamfer_weight: 75.0 (CLI override)"
echo "  - background_depth_iterations: 500 (from config)"
echo "  - general_depth_weight: 0.01 (from config)"
echo ""
echo "======================================================================"
echo ""

# ====================================================================
# CLI OVERRIDES (only these 3 params override config)
# ====================================================================
BATCH_SIZE=4                   # Override config's batch_size=2
PRUNE_INTERVAL=100             # Override config's pruning_interval=700
CHAMFER_WEIGHT=75.0            # Override config's chamfer_weight=50.0

# ====================================================================
# TRAINING COMMAND
# ====================================================================
python train_dynamic_depth.py \
  --batch_size $BATCH_SIZE \
  --chamfer_weight $CHAMFER_WEIGHT \
  --configs "/home/gperezsantamaria/gperezsantamaria2/Egocentric4DGaussians/arguments/HOI4D/default.py" \
  --port 6256 \
  --pruning_interval $PRUNE_INTERVAL \
  --source_path "/home/gperezsantamaria/gperezsantamaria2/Egocentric4DGaussians/data/automatic_data_extraction_testing/with_monst3r/Video1/colmap" \
  --expname video1_parameter_flow_test