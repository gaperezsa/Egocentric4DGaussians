def merge_hparams(args, config, cli_args=None):
    """
    Merge config values into args, but preserve CLI arguments.
    Precedence: CLI args > config > class defaults
    
    Args:
        args: Parsed arguments from argparse
        config: Config object loaded from file
        cli_args: sys.argv (to detect what was explicitly on CLI)
    """
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    # Check if this key was explicitly on CLI
                    was_on_cli = cli_args is not None and any(
                        arg == f"--{key}" for arg in cli_args
                    )
                    # Only apply config value if NOT on CLI
                    if not was_on_cli:
                        setattr(args, key, value)
    return args