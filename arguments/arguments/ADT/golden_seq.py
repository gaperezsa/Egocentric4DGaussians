_base_ = './default.py'
ModelHiddenParams = dict(
    net_width = 128,
)

OptimizationParams = dict(
    iterations = 14000,
    batch_size=4,
    coarse_iterations = 3000
)