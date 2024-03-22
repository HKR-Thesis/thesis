import sys

def measure(target_pid):
    print('Benchmarking not yet implemented for server')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3.x src/benchmarking/jetson_metrics.py <target_pid>")
        sys.exit(1)
    target_pid = int(sys.argv[1])
    measure(target_pid)
