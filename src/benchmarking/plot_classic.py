import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
from scipy.signal import savgol_filter

fieldnames = [
    'Time', 'CPU Util', 'GPU Util', 
    'MEM Util', 'CPU Temp', 'GPU Temp', 
    'CPU Voltage', 'CPU Current', 'GPU Voltage', 
    'GPU Current', 'Total Voltage', 'Total Current', 
    'Average Power Consumption'
]

metric_groups = {
    'util': ['CPU Util', 'GPU Util', 'MEM Util'],
    'temp': ['CPU Temp', 'GPU Temp'],
    'pwr_con': ['CPU Voltage', 'CPU Current', 'GPU Voltage', 'GPU Current', 'Total Voltage', 'Total Current', 'Average Power Consumption'],
}

def plot(csv_path):
    benchmark_data = pd.read_csv(csv_path)
    benchmark_data['Time'] = pd.to_datetime(benchmark_data['Time'])

    window_size = 21
    polyorder = 2
    for metric in fieldnames[1:]:
        if len(benchmark_data) >= window_size:
            benchmark_data[f'{metric} Smooth'] = savgol_filter(benchmark_data[metric], window_length=window_size, polyorder=polyorder)
        else:
            benchmark_data[f'{metric} Smooth'] = benchmark_data[metric]

    _, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 20), sharex=True)

    for i, (metric, group) in enumerate(metric_groups.items()):
        for value in group:
            axs[i].plot(benchmark_data['Time'], benchmark_data[f'{value} Smooth']/1000 if metric is 'pwr_con' else benchmark_data[f'{value} Smooth'], label=value)
        axs[i].legend()
        axs[i].set_ylabel(group[0].split(' ')[-1])
        axs[i].set_title(f'{group[0].split(" ")[-1]} Over Time')

    plt.subplots_adjust(hspace=0.5)

    for ax in axs:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.savefig(f"../../out/plots/{datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_metrics.py <path_to_csv_file>')
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        plot(csv_path)
    except Exception as e:
        print(f"Error: {e}")
