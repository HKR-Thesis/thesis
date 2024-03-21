import subprocess
import sys
import threading

def print_output(stream):
    for line in iter(stream.readline, b''):
        print(line.decode(), end='')

def measure(training_type):
    print("Calling main process")
    main_proc = subprocess.Popen(
        ['python3.10', '-m', 'src.main', training_type],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )
    
    stdout_thread = threading.Thread(target=print_output, args=(main_proc.stdout,))
    stderr_thread = threading.Thread(target=print_output, args=(main_proc.stderr,))
    stdout_thread.start()
    stderr_thread.start()
    
    print("Calling metrics")
    subprocess.Popen(['python3.10', 'src/benchmarking/jetson_metrics.py', str(main_proc.pid)])

    main_proc.wait()
    stdout_thread.join()
    stderr_thread.join()

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        print("Starting measurements")
        training_type = sys.argv[1]
        measure(training_type)
    else:
        print('Usage: python3.10 script.py <training_type>')
