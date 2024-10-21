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
    1. lora
    2. quantization like qlora
    3. 8bit optimizer
5. but the activation memory is scale very quick by batch and sequence length
   - show the harmony figure -> is bottleneck
6. enable gradient checkpointing
    - remain memory is checkpoint value and temporary value
    - the activation memory mainly coming from cross entropy
        - use previous discussion notion
7. solving this problem is huge

## Method

### Fused Linear Cross Entropy

1. explain the cross entropy

2. explain method

### other method like ...

### Liger kernel framework design

## Experimental Results

### paper result

## DEMO

1. show the throught scale by batch size
2. show the ablation study on different method

## Conclusion and Personal Reflection

## Code Related

### Prerequisites

1. Huggingface related stuff