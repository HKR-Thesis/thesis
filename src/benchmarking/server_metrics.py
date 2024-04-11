import sys
import os
import pynvml
import psutil
import csv
import subprocess
import time
import re
from src.benchmarking.plot import fieldnames
from src.util import find_project_root
from datetime import datetime
from pathlib import Path


def get_average_cpu_temperature():
    """
    Retrieves the average CPU temperature over no. cores, specifically for Unix based systems.

    Returns:
        Average CPU temperature
    """
    temps = psutil.sensors_temperatures()
    if "k10temp" in temps:
        core_temps = temps["k10temp"]
        avg_temp = sum(temp.current for temp in core_temps) / len(core_temps)
        return avg_temp
    return "N/A"


def get_cpu_power_consumption_one_shot():
    """
    Retrieves the CPU power consumption in watts using the turbostat command.

    Returns:
        float: The CPU power consumption in watts, or None if an error occurs.
    """
    try:
        result = subprocess.run(
            [
                "sudo",
                "turbostat",  # Assumes turbostat is installed on the underlying system
                "--quiet",
                "--Summary",
                "--show",
                "PkgWatt",
                "sleep",
                "0.1",
            ],
            capture_output=True,
            text=True,
        )
        output = result.stdout
        error = result.stderr

        pkg_watt_match_out = re.search(r"PkgWatt", output)
        pkg_watt_match_err = re.search(r"PkgWatt", error)
        if pkg_watt_match_out:
            pkg_watt = float(output.split()[3])
            return pkg_watt
        elif pkg_watt_match_err:
            pkg_watt = float(error.split()[3])
            return pkg_watt
    except Exception as e:
        print(f"Error reading CPU power consumption: {e}")
    return None


def get_metrics(process, nvml_handle):
    """
    Get various metrics related to the server's performance.

    Args:
        process: A process object representing the server process.
        nvml_handle: A handle to the NVML library.

    Returns:
        A dictionary containing the following metrics:
        - "Time": The current time in the format HH:MM:SS.
        - "CPU Util": The CPU utilization percentage of the server process.
        - "GPU Util": The GPU utilization percentage of the server process.
        - "MEM Util": The memory utilization percentage of the server process.
        - "CPU Temp": The average CPU temperature.
        - "GPU Temp": The GPU temperature.
        - "CPU Power Consumption": The CPU power consumption in watts.
        - "GPU Power Consumption": The GPU power consumption in watts.
    """
    return {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "CPU Util": process.cpu_percent(interval=1.0),
        "GPU Util": pynvml.nvmlDeviceGetUtilizationRates(nvml_handle).gpu,
        "MEM Util": process.memory_percent(),
        "CPU Temp": get_average_cpu_temperature(),
        "GPU Temp": pynvml.nvmlDeviceGetTemperature(
            nvml_handle, pynvml.NVML_TEMPERATURE_GPU
        ),
        "CPU Power Consumption": get_cpu_power_consumption_one_shot(),
        "GPU Power Consumption": pynvml.nvmlDeviceGetPowerUsage(nvml_handle)
        / 1000,  # watts!
    }


def measure(target_pid, training_type):
    """
    Measure server metrics for a given process ID and training type.

    Args:
        target_pid (int): The process ID of the target process.
        training_type (str): The type of training being performed.

    Returns:
        None
    """
    current_file_path = Path(__file__).resolve().parent
    project_root = find_project_root(current_file_path)

    filename = f"{project_root}/out/metrics/server-metrics-{training_type}_{datetime.now().strftime('%Y-%m-%d@%H-%M-%S')}.csv"
    process = psutil.Process(target_pid)

    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        while process.is_running():
            metrics = get_metrics(process, nvml_handle)
            writer.writerow(metrics)
            time.sleep(2.5)

    print(f"Finished collecting metrics - (calling process <{target_pid}> killed)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python3.x src/benchmarking/server_metrics.py <target_pid> <training_type>"
        )
        sys.exit(1)
    target_pid = int(sys.argv[1])
    training_type = sys.argv[2]
    measure(target_pid, training_type)
