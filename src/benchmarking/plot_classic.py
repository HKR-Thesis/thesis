import pandas as pd
import sys
import matplotlib
from matplotlib import pyplot as plt

def plot(csv_path):
    benchmark_data = pd.read_csv(csv_path)
    plt.plot(benchmark_data['CPU Util'], benchmark_data['Time'])
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3.x plot_classic.py <path_to_csv_file>')
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        plot(csv_path)
    except Exception as e:
        print(f"Error: {e}")
