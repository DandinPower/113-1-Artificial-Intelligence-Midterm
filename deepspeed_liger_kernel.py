import os
import torch
import time
import deepspeed

from argparse import ArgumentParser, Namespace
from torch import distributed
from transformers import SchedulerType
from pickle import dump
from deepspeed import comm as dist

from src.model_utils import create_model_by_deepspeed_liger_kernel
from src.optimizer_utils import create_optimizer
from src.ds_utils import get_ds_zero3_infinity_config
from src.utils import set_random_seed, print_rank_0, print_verbose, create_liger_kernel_config_by_args, get_vocab_size, get_dummy_inputs_and_labels, get_snap_shot_name

SNAP_SHOT_DIRS = "snap_shots"
PROFILE_LOG_DIRS = "logs"
RESULT_DIRS = "results"

class DeepSpeedTrainer:
    def __init__(self):
        self.device = None
        self.tokenizer = None 
        self.model = None
        self.optimizer = None
        self.vocab_size = None

    def _set_device(self, local_rank: int) -> None:
        assert local_rank is not None, "local_rank must be provided"
        if local_rank == -1:
            self.device = torch.device("cuda")
        else:
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)

    def _init_distributed(self, local_rank: int) -> None:
        assert local_rank is not None, "local_rank must be provided"
        if local_rank != -1:
            deepspeed.init_distributed("nccl")

    def _add_args(self, args: Namespace) -> Namespace:
        args.global_rank = distributed.get_rank()
        args.train_batch_size = args.per_device_train_batch_size * distributed.get_world_size() * args.gradient_accumulation_steps
        args.train_micro_batch_size_per_gpu = args.per_device_train_batch_size
        return args
    
    def init(self, args: Namespace, verbose: bool=True) -> None:
        self._set_device(args.local_rank)
        self._init_distributed(args.local_rank)
        assert self.device is not None, "device must be set"
        assert args is not None, "args must be provided"

        set_random_seed(args.seed)
        args = self._add_args(args)
        ds_config = get_ds_zero3_infinity_config(args)
        liger_kernel_config = create_liger_kernel_config_by_args(args)
        self.vocab_size = get_vocab_size(args.model_name)

        distributed.barrier()

        print_verbose('[INIT] Create Model', verbose)
        self.model = create_model_by_deepspeed_liger_kernel(ds_config, liger_kernel_config, model_name=args.model_name, gradient_checkpointing=args.gradient_checkpointing, is_flash_attn=args.is_flash_attn)
        print_verbose('[INIT] Model created successfully', verbose)

        print_verbose('[INIT] Create Optimizer', verbose)
        self.optimizer = create_optimizer(self.model, lr=args.learning_rate, weight_decay=args.weight_decay, betas_0=args.beta_0, betas_1=args.beta_1) 
        print_verbose('[INIT] Optimizer created successfully', verbose)
        
        print_verbose('[INIT] DeepSpeed Engine Initialize', verbose)
        self.model, self.optimizer, _, _ = deepspeed.initialize(model=self.model, optimizer=self.optimizer, config=ds_config, dist_init_required=True)
        self.model: deepspeed.DeepSpeedEngine
        print_verbose('[INIT] DeepSpeed Engine Initialized successfully', verbose)

    def train(self, args: Namespace, verbose: bool=True) -> None:
        assert self.device is not None, "device must be set"
        assert self.model is not None, "model must be set"
        assert self.optimizer is not None, "optimizer must be set"
        assert self.vocab_size is not None, "vocab_size must be set"
        assert args is not None, "args must be provided"

        print_verbose("[TRAIN] Start Running training", verbose)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.memory._record_memory_history()
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./{PROFILE_LOG_DIRS}/{get_snap_shot_name(args)}'),
            record_shapes=True,
            with_stack=True,
            profile_memory=True)
        prof.start()
        total_time = 0
        self.model.train()
        for step in range(args.num_train_iterations):
            print_rank_0(f"[TRAIN] Start Running Step: {step}", args.global_rank)
            inputs, labels = get_dummy_inputs_and_labels(args.train_micro_batch_size_per_gpu, args.max_seq_len, self.vocab_size)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            start_time = time.perf_counter()
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss
            self.model.backward(loss)
            self.model.step()
            dist.barrier()
            end_time = time.perf_counter()
            total_time += end_time - start_time
        
        max_memory = torch.cuda.max_memory_allocated()
        iteration_latency = total_time / args.num_train_iterations
        throughput = args.train_batch_size * args.max_seq_len / iteration_latency
        print_rank_0(f"[RESULT] Peak VRAM Usage(per gpu): {max_memory / 1024**2:.2f} MB", args.global_rank)
        print_rank_0(f"[RESULT] Iteration Latency(total): {iteration_latency:.2f} s", args.global_rank)
        print_rank_0(f"[RESULT] Throughput(total): {throughput:.2f} (token/s)", args.global_rank)
        with open(f"./{RESULT_DIRS}/{get_snap_shot_name(args)}.txt", 'w') as f:
            f.write(f"peak_vram_usage_per_gpu(bytes),total_throughput(token/s),total_latency(s)\n")
            f.write(f"{max_memory:.2f},{throughput:.2f},{iteration_latency:.2f}\n")

        snapshot = torch.cuda.memory._snapshot()
        with open(f"./{SNAP_SHOT_DIRS}/{get_snap_shot_name(args)}.pickle", 'wb') as f:
            dump(snapshot, f)

        prof.stop()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank of the process in distributed training.')
    parser.add_argument('--model_name', type=str, help='Model name or path to load from.')
    parser.add_argument('--fp16_loss_scale_window', type=int, help='FP16 loss scale window.')
    parser.add_argument('--fp16_initial_scale_power', type=int, help='FP16 initial scale power.')
    parser.add_argument('--fp16_hysteresis', type=int, help='FP16 hysteresis.')
    parser.add_argument('--reduce_bucket_size', type=float, help='Reduce bucket size.')
    parser.add_argument('--gradient_clipping', type=float, help='Gradient clipping threshold.')
    parser.add_argument("--per_device_train_batch_size", type=int, help="Batch size (per device) for the fake training iteration.")
    parser.add_argument("--num_train_iterations", type=int, help="The number of iterations to train for.")
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Gradient accumulation steps.')
    parser.add_argument("--max_seq_len", type=int, help="The maximum sequence length.")
    parser.add_argument('--learning_rate', type=float, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, help='Weight decay.')
    parser.add_argument('--beta_0', type=float, help='Beta 0.')
    parser.add_argument('--beta_1', type=float, help='Beta 1.')
    parser.add_argument('--sub_group_size', type=int, help='Sub group size.')
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument('--step_per_print', type=int, help='Steps per print.')
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--enable_liger_rope", type=int)
    parser.add_argument("--enable_liger_swiglu", type=int)
    parser.add_argument("--enable_liger_rms", type=int)
    parser.add_argument("--enable_liger_flce", type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--is_flash_attn', action='store_true', help='Enable Flash Attention 2.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    os.makedirs(SNAP_SHOT_DIRS, exist_ok=True)
    os.makedirs(PROFILE_LOG_DIRS, exist_ok=True)
    os.makedirs(RESULT_DIRS, exist_ok=True)

    trainer = DeepSpeedTrainer()
    trainer.init(args)
    trainer.train(args)