import sys
import pynvml
import psutil
import csv
import subprocess
import time
import re
from src.benchmarking.util import fieldnames
from datetime import datetime


def get_average_cpu_temperature():
    """
    Retrieves the average CPU temperature over cores, specifically for Unix based systems.

    Returns:
        Average CPU temperature
    """
    temps = psutil.sensors_temperatures()
    if "coretemp" in temps:
        core_temps = temps["coretemp"]
        avg_temp = sum(temp.current for temp in core_temps) / len(core_temps)
        return avg_temp
    return "N/A"


def get_cpu_power_consumption_one_shot():
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

        pkg_watt_match = re.search(r"PkgWatt:\s*([\d.]+)", output)
        if pkg_watt_match:
            pkg_watt = float(pkg_watt_match.group(1))
            return pkg_watt
    except Exception as e:
        print(f"Error reading CPU power consumption: {e}")
    return None


def get_metrics(process, nvml_handle):
    return {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "CPU Util": process.cpu_percent(interval=1.0),
        "GPU Util": nvml_handle.nvmlDeviceGetUtilizationRates(nvml_handle).gpu,
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
    filename = f"/out/server_measurements/server-metrics-{training_type}_{datetime.now().strftime('%Y-%m-%d@%H-%M-%S')}.csv"
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

    print(f"Finished collecting metrics - (calling process, {target_pid}, killed)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python3.x src/benchmarking/server_metrics.py <target_pid> <training_type>"
        )
        sys.exit(1)
    target_pid = int(sys.argv[1])
    training_type = sys.argv[2]
    measure(target_pid, training_type)
