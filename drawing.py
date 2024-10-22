from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

# Reordered strategies
STRATEGIES_2 = ["No Liger Kernel", "All Liger Kernel"]
STRATEGIES_3 = ["No Liger Kernel", "Liger Kernel w/o FLCE", "All Liger Kernel"]

@dataclass
class Data:
    strategy: str
    batch_size: int
    seq_length: int
    peak_memory: float  # in bytes
    throughput: float    # in token/s

def draw_comparison_with_2_strategy(datas: list[Data]):
    """
    Draw throughput and peak memory comparisons with two strategies.
    """
    # Convert peak_memory to MB
    for data in datas:
        data.peak_memory /= 1024 * 1024  # Bytes to MB

    # Assume all data have the same seq_length
    seq_length = datas[0].seq_length
    print(f"Sequence Length: {seq_length}")

    # Get sorted unique batch sizes
    batch_sizes = sorted(set(data.batch_size for data in datas))
    print(f"Batch Sizes: {batch_sizes}")

    # Create a mapping from (strategy, batch_size) to throughput and peak_memory
    throughput_map = {(data.strategy, data.batch_size): data.throughput for data in datas}
    peak_memory_map = {(data.strategy, data.batch_size): data.peak_memory for data in datas}
    print(f"Throughput Map: {throughput_map}")
    print(f"Peak Memory Map: {peak_memory_map}")

    # Define color mapping for strategies
    strategy_colors = {
        "No Liger Kernel": 'darkorange',      # "No Liger Kernel" as dark orange
        "All Liger Kernel": 'skyblue'         # "All Liger Kernel" as sky blue
    }

    # Number of strategies
    num_strategies = len(STRATEGIES_2)
    # Width of each bar
    bar_width = 0.35
    # Positions of the batch sizes on the x-axis
    x = np.arange(len(batch_sizes))

    # Adjust the figure size to be wider
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

    # ----------------------------
    # Plot Throughput Comparison
    # ----------------------------
    for i, strategy in enumerate(STRATEGIES_2):
        # Get throughput values for each batch size
        throughputs = []
        ooms = []
        for bs in batch_sizes:
            throughput = throughput_map.get((strategy, bs))
            if throughput is not None:
                throughputs.append(throughput)
                ooms.append(False)
            else:
                # Represent OOM as zero and mark as True
                throughputs.append(0)
                ooms.append(True)

        # Calculate positions for each strategy's bars
        offset = (i - num_strategies / 2) * bar_width + bar_width / 2
        positions = x + offset

        # Assign colors based on strategy and OOM status
        colors = ['lightgray' if oom else strategy_colors.get(strategy, 'skyblue') for oom in ooms]

        # Plot bars with the assigned colors
        bars = ax1.bar(positions, throughputs, bar_width, label=strategy, color=colors, edgecolor='black')

        # Annotate OOM bars
        for bar, oom in zip(bars, ooms):
            if oom:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, height + max(throughputs)*0.02, 'OOM',
                         ha='center', va='bottom', color='red', fontsize=9, rotation=90)

    # Set labels and title for throughput
    ax1.set_xlabel('Batch Size', fontsize=14)
    ax1.set_ylabel('Throughput (token/s)', fontsize=14)
    ax1.set_title(f'Llama3.2 1B Throughput Comparison (Sequence Length: {seq_length})', fontsize=16)
    ax1.legend(title='Strategies')
    ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

    # ----------------------------
    # Plot Peak Memory Comparison
    # ----------------------------
    for i, strategy in enumerate(STRATEGIES_2):
        # Get peak memory values for each batch size
        peak_memories = []
        ooms = []
        for bs in batch_sizes:
            peak_memory = peak_memory_map.get((strategy, bs))
            if peak_memory is not None:
                peak_memories.append(peak_memory)
                ooms.append(False)
            else:
                # Represent OOM as zero and mark as True
                peak_memories.append(0)
                ooms.append(True)

        # Calculate positions for each strategy's bars
        offset = (i - num_strategies / 2) * bar_width + bar_width / 2
        positions = x + offset

        # Assign colors based on strategy and OOM status
        colors = ['lightgray' if oom else strategy_colors.get(strategy, 'skyblue') for oom in ooms]

        # Plot bars with the assigned colors
        bars = ax2.bar(positions, peak_memories, bar_width, label=strategy, color=colors, edgecolor='black')

        # Annotate OOM bars
        for bar, oom in zip(bars, ooms):
            if oom:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height + max(peak_memories)*0.02, 'OOM',
                         ha='center', va='bottom', color='red', fontsize=9, rotation=90)

    # Set labels and title for peak memory
    ax2.set_xlabel('Batch Size', fontsize=14)
    ax2.set_ylabel('Peak Memory (MB)', fontsize=14)
    ax2.set_title(f'Llama3.2 1B Peak Memory Comparison (Sequence Length: {seq_length})', fontsize=16)
    ax2.legend(title='Strategies')
    ax2.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

    # Set x-ticks and labels for both subplots
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes, fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes, fontsize=12)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save and show the plot
    plt.savefig("comparison_2_strategies.png", dpi=300)
    # plt.show()

def draw_comparison_with_3_strategy(datas: list[Data]):
    """
    Draw throughput and peak memory comparisons with three strategies.
    The x-axis is batch_size, and there are two side-by-side subplots:
    - Throughput (token/s)
    - Peak Memory (MB)
    Each batch_size has bars for each strategy.
    If there is no data for a strategy, it represents OOM (Out Of Memory).
    """
    # Convert peak_memory to MB
    for data in datas:
        data.peak_memory /= 1024 * 1024  # Bytes to MB

    # Assume all data have the same seq_length
    if not datas:
        print("No data provided.")
        return
    seq_length = datas[0].seq_length
    print(f"Sequence Length: {seq_length}")

    # Get sorted unique batch sizes
    batch_sizes = sorted(set(data.batch_size for data in datas))
    print(f"Batch Sizes: {batch_sizes}")

    # Create a mapping from (strategy, batch_size) to throughput and peak_memory
    throughput_map = {(data.strategy, data.batch_size): data.throughput for data in datas}
    peak_memory_map = {(data.strategy, data.batch_size): data.peak_memory for data in datas}
    print(f"Throughput Map: {throughput_map}")
    print(f"Peak Memory Map: {peak_memory_map}")

    # Define color mapping for strategies
    strategy_colors = {
        "No Liger Kernel": 'darkorange',          # "No Liger Kernel" as dark orange
        "Liger Kernel w/o FLCE": 'green',       # "Liger Kernel w/o FLCE" as sky green
        "All Liger Kernel": 'skyblue'             # "All Liger Kernel" as sky blue
    }

    # Number of strategies
    num_strategies = len(STRATEGIES_3)
    # Width of each bar
    bar_width = 0.25
    # Positions of the batch sizes on the x-axis
    x = np.arange(len(batch_sizes))

    # Adjust the figure size to be wider
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), sharey=False)

    # ----------------------------
    # Plot Throughput Comparison
    # ----------------------------
    for i, strategy in enumerate(STRATEGIES_3):
        # Get throughput values for each batch size
        throughputs = []
        ooms = []
        for bs in batch_sizes:
            throughput = throughput_map.get((strategy, bs))
            if throughput is not None:
                throughputs.append(throughput)
                ooms.append(False)
            else:
                # Represent OOM as zero and mark as True
                throughputs.append(0)
                ooms.append(True)

        # Calculate positions for each strategy's bars
        offset = (i - num_strategies / 2) * bar_width + bar_width / 2
        positions = x + offset

        # Assign colors based on strategy and OOM status
        colors = ['lightgray' if oom else strategy_colors.get(strategy, 'skyblue') for oom in ooms]

        # Plot bars with the assigned colors
        bars = ax1.bar(positions, throughputs, bar_width, label=strategy, color=colors, edgecolor='black')

        # Annotate OOM bars
        for bar, oom in zip(bars, ooms):
            if oom:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2, height + max(throughputs)*0.02, 'OOM',
                         ha='center', va='bottom', color='red', fontsize=9, rotation=90)

    # Set labels and title for throughput
    ax1.set_xlabel('Batch Size', fontsize=16)
    ax1.set_ylabel('Throughput (token/s)', fontsize=16)
    ax1.set_title(f'Llama3.2 1B Throughput Comparison (Sequence Length: {seq_length})', fontsize=18)
    ax1.legend(title='Strategies', fontsize=12)
    ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

    # ----------------------------
    # Plot Peak Memory Comparison
    # ----------------------------
    for i, strategy in enumerate(STRATEGIES_3):
        # Get peak memory values for each batch size
        peak_memories = []
        ooms = []
        for bs in batch_sizes:
            peak_memory = peak_memory_map.get((strategy, bs))
            if peak_memory is not None:
                peak_memories.append(peak_memory)
                ooms.append(False)
            else:
                # Represent OOM as zero and mark as True
                peak_memories.append(0)
                ooms.append(True)

        # Calculate positions for each strategy's bars
        offset = (i - num_strategies / 2) * bar_width + bar_width / 2
        positions = x + offset

        # Assign colors based on strategy and OOM status
        colors = ['lightgray' if oom else strategy_colors.get(strategy, 'skyblue') for oom in ooms]

        # Plot bars with the assigned colors
        bars = ax2.bar(positions, peak_memories, bar_width, label=strategy, color=colors, edgecolor='black')

        # Annotate OOM bars
        for bar, oom in zip(bars, ooms):
            if oom:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height + max(peak_memories)*0.02, 'OOM',
                         ha='center', va='bottom', color='red', fontsize=9, rotation=90)

    # Set labels and title for peak memory
    ax2.set_xlabel('Batch Size', fontsize=16)
    ax2.set_ylabel('Peak Memory (MB)', fontsize=16)
    ax2.set_title(f'Llama3.2 1B Peak Memory Comparison (Sequence Length: {seq_length})', fontsize=18)
    ax2.legend(title='Strategies', fontsize=12)
    ax2.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

    # Set x-ticks and labels for both subplots
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes, fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes, fontsize=14)

    # Adjust layout for better fit
    plt.tight_layout()

    # Save and show the plot
    plt.savefig("comparison_3_strategies.png", dpi=300)
    # plt.show()

def main():
    # Data for two strategies
    datas_2 = [
        Data("No Liger Kernel", 1, 1024, 4121754112.00, 331.25),
        Data("All Liger Kernel", 1, 1024, 3596417536.00, 320.09),
        Data("No Liger Kernel", 4, 1024, 9646049792.00, 1281.24),
        Data("All Liger Kernel", 4, 1024, 3596466688.00, 1266.56),
        Data("No Liger Kernel", 8, 1024, 18382989824.00, 2429.62),
        Data("All Liger Kernel", 8, 1024, 3596532224.00, 2468.04),
        Data("All Liger Kernel", 16, 1024, 4344380416.00, 4354.60),
        Data("All Liger Kernel", 32, 1024, 7097285632.00, 6945.99),
        Data("All Liger Kernel", 64, 1024, 12605193216.00, 9260.56),
        Data("All Liger Kernel", 96, 1024, 18113100800.00, 10244.01),
        Data("All Liger Kernel", 112, 1024, 20867054592.00, 10130.74),
        # You can add more Data instances here, including cases where some strategies might OOM
    ]
    draw_comparison_with_2_strategy(datas_2)

    # Data for three strategies
    datas_3 = [
        Data("No Liger Kernel", 1, 1024, 4121754112.00, 331.25),
        Data("Liger Kernel w/o FLCE", 1, 1024, 4121754112.00, 321.48),
        Data("All Liger Kernel", 1, 1024, 3596417536.00, 320.09),
        Data("No Liger Kernel", 4, 1024, 9646049792.00, 1281.24),
        Data("Liger Kernel w/o FLCE", 4, 1024, 9612495360.00, 1274.67),
        Data("All Liger Kernel", 4, 1024, 3596466688.00, 1266.56),
        Data("No Liger Kernel", 8, 1024, 18382989824.00, 2429.62),
        Data("Liger Kernel w/o FLCE", 8, 1024, 18315880960.00, 2401.27),
        Data("All Liger Kernel", 8, 1024, 3596532224.00, 2468.04),
        Data("All Liger Kernel", 16, 1024, 4344380416.00, 4354.60),
        Data("All Liger Kernel", 32, 1024, 7097285632.00, 6945.99),
        Data("All Liger Kernel", 64, 1024, 12605193216.00, 9260.56),
        Data("All Liger Kernel", 96, 1024, 18113100800.00, 10244.01),
        Data("All Liger Kernel", 112, 1024, 20867054592.00, 10130.74),
        # You can add more Data instances here, including cases where some strategies might OOM
    ]
    draw_comparison_with_3_strategy(datas_3)

if __name__ == "__main__":
    main()
