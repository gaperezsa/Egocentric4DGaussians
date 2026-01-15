ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4],
    defor_depth = 1,
    net_width = 128,
    # ========== Loss Weights (DN-Splatter & Depth) ==========
    general_depth_weight = 1,           # Depth supervision (all depth stages)
    rgb_weight = 50.0,                      # RGB supervision weight
    chamfer_weight = 1.0,                 # Chamfer distance (dynamic objects)
    normal_loss_weight = 0.5,             # Normal regularization (geometric)
    scale_loss_weight = 100.0,               # Scale regularization (disk like encouraging)
    ssim_weight = 0.1,                     # SSIM loss (fine_coloring)
    plane_tv_weight = 0.0001,              # Space-time TV regularization
    # ========== Other Hyperparameters ==========
    time_smoothness_weight = 0.001,
    l1_time_planes = 0.0001,
    render_process=True,
    aria_rotated = True  # Rotate visualizations 90° CW for ADT data (raw Aria orientation)
)

OptimizationParams = dict(
    # ========== Stage Iterations (5-stage pipeline) ==========
    background_depth_iterations = 2000,
    background_RGB_iterations = 2000,
    dynamics_depth_iterations = 5000,
    dynamics_RGB_iterations = 5000,
    fine_iterations = 5000,
    
    # ========== Learning Rates - Position ==========
    batch_size = 2,
    dynamic_position_lr_init = 1e-17,
    dynamic_position_lr_final = 1e-18,
    static_position_lr_init = 1e-14,
    static_position_lr_final = 1e-15,
    
    # ========== Learning Rates - Deformation ==========
    deformation_lr_init = 0.000016,
    deformation_lr_final = 0.0000016,
    
    # ========== Densification & Pruning Strategy ==========
    densify_from_iter = 500,
    densification_interval = 200,
    pruning_interval = 400,
    densify_until_iter = 5_000,
    opacity_reset_interval = 2100,
    scale_pruning_factor = 1.0,
    depth_error_threshold_cm = 20.0,  # Pixels with >20cm error are considered "bad"
    depth_blame_percent = 0.1,  # Fraction of pixels to blame based on depth error
    # dataloader=True,
    # coarse_iterations = 3000,
    # grid_lr_init = 0.0016,
    # grid_lr_final = 16,
    opacity_threshold_coarse = 0.1,
    # opacity_threshold_fine_init = 0.005,
    # opacity_threshold_fine_after = 0.005,
)
