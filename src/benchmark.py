import sys
import threading
from src.util import get_metrics_path, run_process, monitor_subprocess


def measure(training_type):
    benchmark_path = get_metrics_path()

    training_command = [sys.executable, "-u", "-m", "src.training.main", training_type]
    train_proc = run_process(training_command, "Training")

    if benchmark_path is not None:
        benchmark_command = [
            sys.executable,
            "-u",
            benchmark_path,
            str(train_proc.pid),
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


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        measure(sys.argv[1])
    else:
        print(f"Usage: {sys.executable} benchmark.py <training_type>")
