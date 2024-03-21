import subprocess, sys

def measure(training_type):
    main_proc = subprocess.Popen(
        ['python3.10', '-m', 'src.train', training_type],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    subprocess.run(['python3.10', 'src/benchmarking/jetson_metrics.py', str(main_proc.pid)])

    stdout, stderr = main_proc.communicate()
    print(f'Main Output: {stdout.decode()}')
    print(f'Main Errors: {stderr.decode()}')

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        training_type = sys.argv[1]
        measure(training_type)
    else:
        print(f'Usage: python3.x src.benchmarking.measure <training_type>')