from argparse import Namespace


def get_ds_zero3_infinity_config(args: Namespace) -> dict:
    ZERO_STAGE = 3
    OFFLOAD_DEVICE = "cpu"

    return {
        "steps_per_print": args.step_per_print,
        "train_batch_size": args.train_batch_size,
        "train_micro_batch_size_per_gpu": args.train_micro_batch_size_per_gpu,
        "gradient_clipping": 0.9,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "fp16": {
            "enabled": True,
            "loss_scale_window": args.fp16_loss_scale_window,
            "initial_scale_power": args.fp16_initial_scale_power,
            "hysteresis": args.fp16_hysteresis,
        },
        "zero_optimization": {
            "stage": ZERO_STAGE,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "offload_param": {
                "device": OFFLOAD_DEVICE,
                "pin_memory": True,
            },
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
                "fast_init": False
            },
            "load_from_fp32_weights": False,
            "stage3_param_persistence_threshold": 0,
            "stage3_max_live_parameters": 0,
            "stage3_prefetch_bucket_size": 0,
            "sub_group_size": args.sub_group_size,
            "memory_efficient_linear": True,
            "round_robin_gradients": False,
            "reduce_bucket_size": args.reduce_bucket_size,
        },
    }
