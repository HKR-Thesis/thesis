import sys
import threading
import argparse
from src.util import get_metrics_path, run_process, monitor_subprocess


def measure(training_type):
    benchmark_path = get_metrics_path()

    training_command = [
        sys.executable,
        "-u",
        "-m",
        "src.training.main",
        "--train",
        training_type,
    ]
    train_proc = run_process(training_command, "Training")

    if benchmark_path is not None:
        benchmark_command = [
            sys.executable,
            "-m",
            "src.benchmarking.jetson_metrics",
            "--pid",
            str(train_proc.pid),
            "--train",
            training_type,
        ]
        benchmark_proc = run_process(benchmark_command, "Benchmark")

        monitor_thread = threading.Thread(
            target=monitor_subprocess, args=(train_proc, benchmark_proc)
        )
        monitor_thread.start()

        monitor_thread.join()
        benchmark_proc.wait()
    else:
        train_proc.wait()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run training and optionally benchmark."
    )
    parser.add_argument(
        "--train", type=str, required=True, help="Specify the type of training"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    measure(args.train)
