import subprocess
import platform
import sys
import threading

benchmark_scripts = {
    'embedded': 'src/benchmarking/jetson_metrics.py',
    'server': 'src/benchmarking/server_metrics.py'
}

def print_pipe(stream, prefix=""):
    while True:
        line = stream.readline()
        if not line:
            break
        print(f"{prefix}: {line.strip()}")

def monitor_subprocess(proc, benchmark_proc=None):
    proc.wait()
    if proc.poll() is not None and benchmark_proc is not None:
        if benchmark_proc.poll() is None:
            benchmark_proc.terminate()
            
def get_metrics_path() -> str:
    uname = platform.uname()
    if uname.system is 'Linux' and 'tegra' in uname.release:
        return benchmark_scripts['embedded']
    elif uname.system is 'Windows':
        return benchmark_scripts['server']
    raise ValueError('System benchmarking not supported')

def measure(training_type):
    train_proc = subprocess.Popen(
        ['python3.10', '-u', '-m', 'src.training.train', training_type],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    threading.Thread(target=print_pipe, args=(train_proc.stdout, "Training Output"), daemon=True).start()
    threading.Thread(target=print_pipe, args=(train_proc.stderr, "Training Error"), daemon=True).start()

    benchmark_proc = subprocess.Popen(
        ['python3.10', '-u', get_metrics_path(), str(train_proc.pid)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    threading.Thread(target=print_pipe, args=(benchmark_proc.stdout, "Benchmark Output"), daemon=True).start()
    threading.Thread(target=print_pipe, args=(benchmark_proc.stderr, "Benchmark Error"), daemon=True).start()
    monitor_thread = threading.Thread(target=monitor_subprocess, args=(train_proc, benchmark_proc), daemon=True)
    
    monitor_thread.start()
    monitor_thread.join()
    benchmark_proc.wait()

if len(sys.argv) >= 2:
    measure(sys.argv[1])
else:
    print('Usage: python3.10 script_name.py <training_type>')