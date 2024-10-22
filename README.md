# Study Report of Liger-Kernel

This is a study report of [Liger-Kernel](https://github.com/linkedin/Liger-Kernel/tree/main), the code part of this repo provides a different aspect demonstration of the Liger-Kernel. 

## Report

### Title and Authors

- Title: Liger Kernel: Efficient Triton Kernels for LLM Training

- Authors: Pin-Lun Hsu, Yun Dai, Vignesh Kothapalli (**LinkedIn Inc.**)

- Published as preprint Paper (2024/10/14)

## Background and Motivation

1. current LLM (Causal Language Model) model architecture
2. how to train LLM model (cross entropy)
3. The Memory usage during training
    1. static memory
    2. activation memory
4. static memory part can be optimized like 
    1. fp16 mixed precision training
    1. lora
    2. quantization like qlora for model weight, 8bit optimizer for optimizer states
    3. deepspeed zero, fsdp for static memory partition
    3. deepspeed cpu offloading
5. but the activation memory is scale very quick by batch and sequence length
   - show the harmony figure -> is bottleneck
    6. enable gradient checkpointing
        - remain memory is checkpoint value and temporary value
        - the activation memory mainly coming from cross entropy
            - use previous discussion notion
7. solving the memory bottleneck is huge 
    1. showing the actual memory usage after enable cpu offloading / gradient checkpoint -> cross entropy peak
    2. compare batch1 and batch4 and batch 8
8. also the paper address the gpu kernel utilization part
    1. reference to section 2 to introduce kernel optimization problem
        1. because in pytorch eager execution mode, we can ease to develop, but the kernel is not fully utilized
        2. showing the lecture graph to show the problem that the gpu kernel is not fully utilized
        3. there have performance improvement space by optimize in kernel level
    2. explain what is operation fusion optimization
        1. like flashattention, unsloth and their method
    3. using the triton, they can optimize the kernel level fully in python, not in c++ cuda level, find a trade-off between performance and development easiness
        - show the lecture reason to use triton
    
    

### Method

#### Fused Linear Cross Entropy

1. explain the cross entropy

2. explain method

#### other method like ...

##### RMS Norm

1. explain the rms normalization forward and backward
2. explain the method

#### Liger kernel framework design (implement/api design)

### Experimental Results

#### correctness test 


### DEMO

1. show the memory usage by batch size
1. show the throught scale by batch size
2. show the ablation study on different method

### Conclusion and Personal Reflection

### Appendix and reference

1. pytorch profiler and tensorboard plugin
2. deepspeed zero, fsdp
3. deepspeed offloading
4. gradient checkpointing
5. lora, qlora
6. 8bit optimizer
7. triton
8. sympy for forward and automatically backward
9. flashattention
10. unsloth.ai 
11. nvidia mixed precision training




## Code Related

### Prerequisites

1. Huggingface login related stuff

pip requirements freeze and install

run and tensorboard profiling
run and pytorch memory snapshots
