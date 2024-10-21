import time
import torch

from dataclasses import dataclass
from typing import Tuple
from pickle import dump
from torch.optim.adamw import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import apply_liger_kernel_to_llama


MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda"

BATCH_SIZE = 1
SEQ_LENGTH = 128
ITERATIONS = 3

ENABLE_ROPE = False
ENABLE_SWIGLU = False
ENABLE_RMS = False
ENABLE_FLCE = False
# ENABLE_ROPE = True
# ENABLE_SWIGLU = True
# ENABLE_RMS = True
# ENABLE_FLCE = True

SNAPSHOT_PATH = f'bs{BATCH_SIZE}_seq{SEQ_LENGTH}_rope{int(ENABLE_ROPE)}_silu{int(ENABLE_SWIGLU)}_rms{int(ENABLE_RMS)}_flce{int(ENABLE_FLCE)}.pickle'


@dataclass
class LigerKernelConfig:
    enable_rope_optimization: bool
    enable_swiglu_optimization: bool
    enable_rms_norm_optimization: bool
    enable_fused_linear_cross_entropy: bool

@dataclass
class HyperParameterConfig:
    batch_size: int
    seq_length: int
    vocab_size: int
    iterations: int

def get_llama_model_by_liger_kernel_config(config: LigerKernelConfig, model_name: str) -> torch.nn.Module:
    apply_liger_kernel_to_llama(
        rope=config.enable_rope_optimization,
        swiglu=config.enable_swiglu_optimization,
        cross_entropy=not(config.enable_fused_linear_cross_entropy),
        fused_linear_cross_entropy=config.enable_fused_linear_cross_entropy,
        rms_norm=config.enable_rms_norm_optimization
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
    model.gradient_checkpointing_enable()
    return model

def get_vocab_size(model_name: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    return len(vocab)

def get_dummy_inputs_and_labels(config: HyperParameterConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_length))
    labels = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_length))
    return inputs, labels

def main():
    liger_kernel_config = LigerKernelConfig(
        enable_rope_optimization=ENABLE_ROPE,
        enable_swiglu_optimization=ENABLE_SWIGLU,
        enable_rms_norm_optimization=ENABLE_RMS,
        enable_fused_linear_cross_entropy=ENABLE_FLCE
    )
    model = get_llama_model_by_liger_kernel_config(liger_kernel_config, MODEL_NAME)
    vocab_size = get_vocab_size(MODEL_NAME)

    hyper_parameter_config = HyperParameterConfig(
        batch_size=BATCH_SIZE,
        seq_length=SEQ_LENGTH,
        vocab_size=vocab_size,
        iterations=ITERATIONS
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history()

    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    total_time = 0
    for _ in range(hyper_parameter_config.iterations):        
        model.train()
        optimizer.zero_grad()
        inputs, labels = get_dummy_inputs_and_labels(hyper_parameter_config)
        inputs = inputs.cuda()
        labels = labels.cuda()
        start_time = time.perf_counter()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        end_time = time.perf_counter()
        total_time += end_time - start_time
    
    max_memory = torch.cuda.max_memory_allocated()
    iteration_latency = total_time / hyper_parameter_config.iterations
    throughput = hyper_parameter_config.batch_size * hyper_parameter_config.seq_length / iteration_latency
    print(f"Peak VRAM Usage: {max_memory / 1024**2:.2f} MB")
    print(f"Iteration Latency: {iteration_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} (token/s)")

    snapshot = torch.cuda.memory._snapshot()
    with open(SNAPSHOT_PATH, 'wb') as f:
        dump(snapshot, f)

if __name__ == "__main__":
    main()