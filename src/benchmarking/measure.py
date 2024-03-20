import subprocess
import time

def main():
    main_proc = subprocess.Popen(
        ['python3.10', '-m', 'src.main'],
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    subprocess.run(['python3.10', 'src/benchmarking/jetson_metrics.py', str(main_proc.pid)])

    stdout, stderr = main_proc.communicate()
    print(f'Main Output: {stdout.decode()}')
    print(f'Main Errors: {stderr.decode()}')

if __name__ == '__main__':
    main()