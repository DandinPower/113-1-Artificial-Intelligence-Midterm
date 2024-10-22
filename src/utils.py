import random
import numpy as np
import torch
from torch.distributed import all_reduce, ReduceOp, get_world_size
from transformers import set_seed, AutoTokenizer
from typing import Tuple
from dataclasses import dataclass
from argparse import Namespace

@dataclass
class LigerKernelConfig:
    enable_rope_optimization: bool
    enable_swiglu_optimization: bool
    enable_rms_norm_optimization: bool
    enable_fused_linear_cross_entropy: bool

def create_liger_kernel_config_by_args(args: Namespace) -> LigerKernelConfig:
    assert args.enable_liger_rope is not None
    assert args.enable_liger_swiglu is not None
    assert args.enable_liger_rms is not None
    assert args.enable_liger_flce is not None
    return LigerKernelConfig(
        enable_rope_optimization=bool(args.enable_liger_rope),
        enable_swiglu_optimization=bool(args.enable_liger_swiglu),
        enable_rms_norm_optimization=bool(args.enable_liger_rms),
        enable_fused_linear_cross_entropy=bool(args.enable_liger_flce)
    )

def get_vocab_size(model_name: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    return len(vocab)

def get_dummy_inputs_and_labels(batch_size: int, max_seq_length: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, vocab_size, (batch_size, max_seq_length))
    labels = torch.randint(0, vocab_size, (batch_size, max_seq_length))
    return inputs, labels

def set_random_seed(seed):
    assert seed is not None, "seed must be provided"
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_rank_0(msg: str, rank: int) -> None:
    assert rank is not None, "rank must be provided"
    if rank == 0:
        print(msg)

def print_verbose(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)

def get_snap_shot_name(args: Namespace) -> str:
    return f"bs{args.train_batch_size}_seq{args.max_seq_len}_liger_rope{args.enable_liger_rope}_silu{args.enable_liger_swiglu}_rms{args.enable_liger_rms}_flce{args.enable_liger_flce}"