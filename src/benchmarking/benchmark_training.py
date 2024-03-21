import subprocess, sys

def measure(training_type):
    print("Calling main process")
    main_proc = subprocess.Popen(
        ['python3.10', '-m', 'src.main', training_type],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    print("Calling metrics process")
    subprocess.run(['python3.10', 'src/benchmarking/jetson_metrics.py', str(main_proc.pid)])

    stdout, stderr = main_proc.communicate()
    print(f'Main Output: {stdout.decode()}')
    print(f'Main Errors: {stderr.decode()}')

if len(sys.argv) >= 2:
    print("Starting measurements")
    training_type = sys.argv[1]
    measure(training_type)
else:
    print(f'Usage: python3.x src.benchmarking.benchmark_training <training_type>')