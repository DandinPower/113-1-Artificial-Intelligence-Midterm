from dataclasses import dataclass
import matplotlib.pyplot as plt

# STRATEGIES = ["All Liger Kernel", "All Liger Kernel w/o FLCE", "No Liger Kernel"]
STRATEGIES = ["All Liger Kernel", "No Liger Kernel"]

@dataclass
class Data:
    strategy: str
    batch_size: int
    seq_length: int
    peak_memory: float
    throughput: float

def draw_throughput_comparison(datas: list[Data]):
    """
    Draw throughput comparison in fixed seq_length with different batch_size and strategies. The x-axis is batch_size, the y-axis is throughput. and each batch_size has 2 bars strategies. if there is no data for a strategy, the bar is empty, which represents OOM (Out Of Memory).
    """
    # convert peak_memory to MB


def main():
    datas = [
        Data("All Liger Kernel", 1, 1024, 1.0, 1024.0),
        Data("No Liger Kernel", 1, 1024, 2.0, 2048.0),
    ]
if __name__ == "__main__":
    main()