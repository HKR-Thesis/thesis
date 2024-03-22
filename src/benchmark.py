import sys
from src.util import get_metrics_path, run_process

import sys

def measure(training_type):
    """
    Measure the training and benchmark the results.

    Args:
        training_type (str): The type of training to perform.

    Returns:
        None
    """
    benchmark_path = get_metrics_path()

    training_command = [sys.executable, '-u', '-m', 'src.training.train', training_type]
    train_proc = run_process(training_command, "Training")

    train_proc.wait()

    if benchmark_path is not None:
        benchmark_command = [sys.executable, '-u', benchmark_path, str(train_proc.pid)]
        benchmark_proc = run_process(benchmark_command, "Benchmark")
        benchmark_proc.wait()

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        measure(sys.argv[1])
    else:
        print(f'Usage: {sys.executable} benchmark.py <training_type>')