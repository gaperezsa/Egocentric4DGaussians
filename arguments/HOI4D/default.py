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
    plane_tv_weight = 100,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    render_process=True
)
OptimizationParams = dict(
    # dataloader=True,
    iterations = 14_000,
    batch_size=2,
    coarse_iterations = 3000,
    densify_until_iter = 4_000,
    opacity_reset_interval = 300000,
    dynamic_position_lr_init = 1e-11,
    dynamic_position_lr_final = 1e-13,
    static_position_lr_init = 1e-14,
    static_position_lr_final = 1e-15,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.000016
    # grid_lr_init = 0.0016,
    # grid_lr_final = 16,
    # opacity_threshold_coarse = 0.005,
    # opacity_threshold_fine_init = 0.005,
    # opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)