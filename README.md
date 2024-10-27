# Liger Kernel Study and Demo

This repository contains code, experiments, and a report for a study on the [Liger Kernel](https://github.com/linkedin/Liger-Kernel) and its application in memory-efficient training of Large Language Models (LLMs). The report analyzes the Liger Kernel's techniques, including fused kernel operations and chunking strategies, and demonstrates its impact on memory usage and throughput.

## Repository Structure

```
.
├── .gitignore                      # Git ignore file
├── all_gpu_liger_kernel.py         # Script for single GPU Liger Kernel experiment
├── deepspeed_liger_kernel.py       # Script for DeepSpeed CPU Offloading, CPU Adam, Gradient Checkpointing and Liger Kernel experiment using DeepSpeed
├── drawing.py                      # Script for generating comparison plots
├── requirements.txt                # Python package requirements
├── README.md                       # This README file
├── run_ds.sh                       # Shell script for running the DeepSpeed experiment
├── src                             # Source code directory for DeepSpeed experiment
│   ├── __init__.py
│   ├── ds_utils.py
│   ├── model_utils.py
│   ├── optimizer_utils.py
│   └── utils.py
└── doc                             # Report and slides directory
    ├── image                       # Images used in the report
    ├── report.md                   # Report in markdown format
    ├── group_10_midterm.docx       # Report in Word format
    ├── group_10_midterm.pdf        # Report in PDF format
    ├── group_10_liger_kernel.pptx  # Presentation slides
    └── group_10_liger_kernel.pdf   # Presentation slides in PDF format
```

## Running the Experiments

### Prerequisites

1. **Python Environment:**  Ensure you have a Python environment with the required packages installed. You can create a virtual environment and install the dependencies using:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Hugging Face Hub:**  You'll need to be logged in to the Hugging Face Hub to download the Llama model. Follow the instructions at [huggingface.co](https://huggingface.co) to create an account and obtain an access token. Then, configure your Hugging Face access token:

   ```bash
   huggingface-cli login
   ```

### Single GPU Experiment

To run the single GPU experiment using Liger Kernel:

```bash
python all_gpu_liger_kernel.py
```

This script will run a fake training iteration with Liger Kernel enabled and output peak memory usage, iteration latency, and throughput. It will also save a memory snapshot.  Modify the `ENABLE_ROPE`, `ENABLE_SWIGLU`, `ENABLE_RMS`, and `ENABLE_FLCE` variables in the script to control which Liger Kernel optimizations are applied.


### CPU Offloading, CPU Adam, Gradient Checkpointing, and Liger Kernel Experiment

This experiment uses the DeepSpeed library to offload all static memory into CPU memory, and also use CPU Adam to offload the optimizer operations to the CPU. The experiment also uses gradient checkpointing to reduce activation memory usage. After all these optimizations, the impact of the Liger Kernel can be observed.

To run the experiment:

1. Modify the parameters in `run_ds.sh`, such as `NUM_GPUS`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `MAX_SEQ_LENGTH`, etc., to adjust the experiment settings.
3. Run the script:

   ```bash
   bash run_ds.sh
   ```

This script will run a fake training iteration using DeepSpeed and Liger Kernel. The results (peak memory, throughput, latency) will be printed to the console and saved in `results/<snapshot_name>.txt`. A memory snapshot will also be saved to the `snap_shots` directory, and TensorBoard profiling logs will be written to the `logs` directory.

### Viewing Profiler Logs

To visualize the profiler logs, use TensorBoard:

```bash
tensorboard --logdir=logs/
```

### Viewing Memory Snapshots

Use the PyTorch memory visualizer as described in [pytorch.org/memory_viz](pytorch.org/memory_viz) to analyze the memory snapshots saved in the `snap_shots` directory.


### Generating Comparison Plots

The script `drawing.py` generates comparison plots for throughput and peak memory usage between different strategies (e.g., with and without Liger Kernel).  The data for these plots are hardcoded within the `drawing.py` script.  You'll need to update these values with the results of your experiments to generate accurate comparison plots.  Run the script:

```bash
python drawing.py
```

The plots will be saved in the current directory as PNG files.


## Report and Slides

The `doc` directory contains the report (`report.md`, `group_10_midterm.docx`, `group_10_midterm.pdf`) and presentation slides (`group_10_liger_kernel.pptx`) summarizing the study on the Liger Kernel. So if you want to know more about the study, please refer to these documents.