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
    general_depth_weight = 0.01,           # Depth supervision (all depth stages)
    rgb_weight = 10.0,                      # RGB supervision weight
    chamfer_weight = 50.0,                 # Chamfer distance (dynamic objects)
    normal_loss_weight = 0.05,             # Normal regularization (geometric)
    scale_loss_weight = 1.0,               # Scale regularization (disk like encouraging)
    ssim_weight = 0.1,                     # SSIM loss (fine_coloring)
    plane_tv_weight = 0.0001,              # Space-time TV regularization
    # ========== Other Hyperparameters ==========
    time_smoothness_weight = 0.001,
    l1_time_planes = 0.0001,
    render_process=True
)

OptimizationParams = dict(
    # ========== Stage Iterations (5-stage pipeline) ==========
    background_depth_iterations = 500,
    background_RGB_iterations = 300,
    dynamics_depth_iterations = 200,
    dynamics_RGB_iterations = 200,
    fine_iterations = 300,
    
    # ========== Learning Rates - Position ==========
    batch_size = 2,
    dynamic_position_lr_init = 1e-16,
    dynamic_position_lr_final = 1e-17,
    static_position_lr_init = 1e-14,
    static_position_lr_final = 1e-15,
    
    # ========== Learning Rates - Deformation ==========
    deformation_lr_init = 0.000016,
    deformation_lr_final = 0.0000016,
    
    # ========== Densification & Pruning Strategy ==========
    densification_interval = 400,
    pruning_interval = 700,
    densify_until_iter = 5_000,
    opacity_reset_interval = 300000,
    scale_pruning_factor = 1.0
    # dataloader=True,
    # coarse_iterations = 3000,
    # grid_lr_init = 0.0016,
    # grid_lr_final = 16,
    # opacity_threshold_coarse = 0.005,
    # opacity_threshold_fine_init = 0.005,
    # opacity_threshold_fine_after = 0.005,
)
