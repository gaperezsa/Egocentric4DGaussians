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
    # ========== K-Planes Temporal Regulation ==========
    time_smoothness_weight = 0.5,
    l1_time_planes = 0.001,
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
    static_position_lr_init = 1e-15,
    static_position_lr_final = 1e-16,

    # ========== Learning Rates - Deformation ==========
    deformation_lr_init = 0.0016,
    deformation_lr_final = 0.00016,

    grid_lr_init = 0.016,
    grid_lr_final = 0.0016,

    # ========== Densification & Pruning Strategy ==========
    densify_from_iter = 500,
    densification_interval = 400,
    pruning_interval = 300,
    densify_until_iter = 3_000,
    opacity_reset_interval = 2100,
    scale_pruning_factor = 1.0,

    # ========== Gaussian Splitting Control ==========
    split_N = 2,
    split_scale_factor = 3.0,

    # ========== All-Dynamic Fine Coloring ==========
    all_dynamic_on_fine = True,

    # ========== Importance Sampling for Fine Coloring ==========
    importance_sampling_fine = True,

    # ========== Border-crop masking (fine_coloring losses only) ==========
    # Exclude the artifact border pixels found in Video2 GT images.
    # 4px black/near-black column on the left; 1px dark row at the bottom.
    # Tensor shapes are NOT changed — the mask is applied only inside loss functions.
    border_crop_left   = 4,
    border_crop_bottom = 1,
)


# ========== ModelParams (Dataset-specific) ==========
ModelParams = dict(
    # HOI4D Phase Filtering: Use only static frames for background training
    use_phase_filtering = True,
    white_background = False,
)
