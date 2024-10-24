from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

# Define the Data class
@dataclass
class Data:
    strategy: str
    batch_size: int
    seq_length: int
    peak_memory: float  # in bytes
    throughput: float   # in token/s

def draw_comparison(
    datas: List[Data],
    strategies: List[str],
    x_axis: str,
    fixed_value: Optional[int] = None,
    fixed_type: Optional[str] = None,
    output_filename: str = 'comparison.png',
    strategy_colors: Optional[dict] = None
):
    """
    Draw throughput and peak memory comparisons for given strategies.
    Can handle variable or fixed batch sizes/sequence lengths.

    Parameters:
    - datas: List of Data instances.
    - strategies: List of strategies to compare.
    - x_axis: Variable for the x-axis ('batch_size' or 'seq_length').
    - fixed_value: Value to fix for the other variable (if any).
    - fixed_type: The variable type that is fixed ('batch_size' or 'seq_length').
    - output_filename: Filename to save the plot.
    - strategy_colors: Optional dictionary mapping strategies to colors.
    """
    # Filter data if a fixed value is provided
    if fixed_value is not None and fixed_type is not None:
        datas = [data for data in datas if getattr(data, fixed_type) == fixed_value]
        if not datas:
            print(f"No data available for {fixed_type} = {fixed_value}.")
            return

    if not datas:
        print("No data provided.")
        return

    # Convert peak_memory to MB
    for data in datas:
        data.peak_memory /= 1024 * 1024  # Convert Bytes to MB

    # Get unique x-axis values
    x_values = sorted(set(getattr(data, x_axis) for data in datas))
    print(f"{x_axis.replace('_', ' ').title()}s: {x_values}")

    # Get fixed variable value for titles
    if fixed_value is not None and fixed_type is not None:
        fixed_var_value = fixed_value
    else:
        # Assume the first data point's fixed variable
        fixed_var = 'seq_length' if x_axis == 'batch_size' else 'batch_size'
        fixed_var_value = getattr(datas[0], fixed_var)

    # Create mappings for throughput and peak memory
    throughput_map = {(data.strategy, getattr(data, x_axis)): data.throughput for data in datas}
    peak_memory_map = {(data.strategy, getattr(data, x_axis)): data.peak_memory for data in datas}

    # Set default strategy colors if not provided
    if strategy_colors is None:
        default_colors = ['darkorange', 'green', 'skyblue', 'purple', 'red', 'cyan']
        strategy_colors = {strategy: default_colors[i % len(default_colors)] for i, strategy in enumerate(strategies)}

    # Number of strategies
    num_strategies = len(strategies)
    # Width of each bar
    bar_width = 0.8 / num_strategies
    # Positions on x-axis
    x = np.arange(len(x_values))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

    # Plotting function for both throughput and peak memory
    def plot_metric(ax, metric_map, ylabel, title):
        for i, strategy in enumerate(strategies):
            # Get metric values and OOM flags
            metrics = []
            ooms = []
            for xv in x_values:
                metric = metric_map.get((strategy, xv))
                if metric is not None:
                    metrics.append(metric)
                    ooms.append(False)
                else:
                    metrics.append(0)
                    ooms.append(True)

            # Calculate bar positions
            offset = (i - num_strategies / 2) * bar_width + bar_width / 2
            positions = x + offset

            # Assign colors
            colors = ['lightgray' if oom else strategy_colors.get(strategy, 'skyblue') for oom in ooms]

            # Plot bars
            bars = ax.bar(positions, metrics, bar_width, label=strategy, color=colors, edgecolor='black')

            # Annotate OOM bars
            for bar, oom in zip(bars, ooms):
                if oom:
                    height = bar.get_height()
                    y_offset = max(metrics) * 0.02 if max(metrics) > 0 else 1
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + y_offset,
                        'OOM',
                        ha='center',
                        va='bottom',
                        color='red',
                        fontsize=9,
                        rotation=90
                    )

        # Set labels and title
        ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        if fixed_value is not None and fixed_type is not None:
            ax.set_title(f'Llama3.2 1B {title} ({fixed_type.replace("_", " ").title()}: {fixed_value})', fontsize=16)
        else:
            ax.set_title(f'Llama3.2 1B {title}', fontsize=16)
        ax.legend(title='Strategies')
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_values, fontsize=12, rotation=45)

    # Plot throughput
    plot_metric(ax1, throughput_map, 'Throughput (token/s)', 'Throughput Comparison')

    # Plot peak memory
    plot_metric(ax2, peak_memory_map, 'Peak Memory (MB)', 'Peak Memory Comparison')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    # plt.show()

def main():
    # Strategies for comparison
    strategies_2 = ["No Liger Kernel", "All Liger Kernel"]
    strategies_3 = ["No Liger Kernel", "Liger Kernel w/o FLCE", "All Liger Kernel"]

    strategy_colors = {
        "No Liger Kernel": 'darkorange',          # "No Liger Kernel" as dark orange
        "Liger Kernel w/o FLCE": 'green',       # "Liger Kernel w/o FLCE" as sky green
        "All Liger Kernel": 'skyblue'             # "All Liger Kernel" as sky blue
    }

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
    ]
    # Plot comparison for two strategies varying batch size
    draw_comparison(
        datas=datas_2,
        strategies=strategies_2,
        x_axis='batch_size',
        output_filename='comparison_2_strategies.png',
        strategy_colors=strategy_colors
    )

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
    ]
    # Plot comparison for three strategies varying batch size
    draw_comparison(
        datas=datas_3,
        strategies=strategies_3,
        x_axis='batch_size',
        output_filename='comparison_3_strategies.png',
        strategy_colors=strategy_colors
    )

    # Data for fixed batch size with two strategies
    fixed_batch_datas_2 = [
        Data("No Liger Kernel", 4, 512, 5275261440.00, 652),
        Data("All Liger Kernel", 4, 512, 3596302848.00, 632.81),
        Data("No Liger Kernel", 4, 1024, 9646049792.00, 1281.24),
        Data("All Liger Kernel", 4, 1024, 3596466688.00, 1266.56),
        Data("No Liger Kernel", 4, 2048, 18387626496.00, 2445.88),
        Data("All Liger Kernel", 4, 2048, 3596794368.00, 2440.63),
        Data("All Liger Kernel", 4, 4096, 4345191424.00, 4329.08),
        Data("All Liger Kernel", 4, 8192, 7099177984.00, 6433.85),
        Data("All Liger Kernel", 4, 16384, 12609248256.00, 7167.41),
        Data("All Liger Kernel", 4, 24576, 18120236032.00, 6709.56),
    ]
    # Plot comparison for fixed batch size with two strategies
    fixed_batch_size = 4
    draw_comparison(
        datas=fixed_batch_datas_2,
        strategies=strategies_2,
        x_axis='seq_length',
        fixed_value=fixed_batch_size,
        fixed_type='batch_size',
        output_filename=f"comparison_fixed_batch_{fixed_batch_size}_2_strategies.png",
        strategy_colors=strategy_colors
    )

    # Data for fixed batch size with three strategies
    fixed_batch_datas_3 = [
        Data("No Liger Kernel", 4, 512, 5275261440.00, 652),
        Data("Liger Kernel w/o FLCE", 4, 512, 5258484224.00, 643.07),
        Data("All Liger Kernel", 4, 512, 3596302848.00, 632.81),
        Data("No Liger Kernel", 4, 1024, 9646049792.00, 1281.24),
        Data("Liger Kernel w/o FLCE", 4, 1024, 9612495360.00, 1274.67),
        Data("All Liger Kernel", 4, 1024, 3596466688.00, 1266.56),
        Data("No Liger Kernel", 4, 2048, 18387626496.00, 2445.88),
        Data("Liger Kernel w/o FLCE", 4, 2048, 18320517632.00, 2430.19),
        Data("All Liger Kernel", 4, 2048, 3596794368.00, 2440.63),
        Data("All Liger Kernel", 4, 4096, 4345191424.00, 4329.08),
        Data("All Liger Kernel", 4, 8192, 7099177984.00, 6433.85),
        Data("All Liger Kernel", 4, 16384, 12609248256.00, 7167.41),
        Data("All Liger Kernel", 4, 24576, 18120236032.00, 6709.56),
    ]
    # Plot comparison for fixed batch size with three strategies
    draw_comparison(
        datas=fixed_batch_datas_3,
        strategies=strategies_3,
        x_axis='seq_length',
        fixed_value=fixed_batch_size,
        fixed_type='batch_size',
        output_filename=f"comparison_fixed_batch_{fixed_batch_size}_3_strategies.png",
        strategy_colors=strategy_colors
    )

if __name__ == "__main__":
    main()
