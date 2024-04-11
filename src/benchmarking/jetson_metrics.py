import csv
import time
import argparse
from datetime import datetime
from src.benchmarking.plot import fieldnames
from jtop import jtop


def get_metrics(jetson) -> dict:
    """
    Get the metrics of the Jetson device.

    Parameters:
    - jetson: The Jetson device object.

    Returns:
    - metrics: A dictionary containing the following metrics:
        - "Time": The current time in the format HH:MM:SS.
        - "CPU Util": The CPU utilization percentage.
        - "GPU Util": The GPU utilization percentage.
        - "MEM Util": The memory utilization percentage.
        - "CPU Temp": The CPU temperature in degrees Celsius.
        - "GPU Temp": The GPU temperature in degrees Celsius.
        - "CPU Power Consumption": The CPU power consumption in watts.
        - "GPU Power Consumption": The GPU power consumption in milliwatts or watts.
    """
    return {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "CPU Util": jetson.cpu["total"]["user"],
        "GPU Util": jetson.gpu["load"],
        "MEM Util": jetson.memory["RAM"]["used"] / jetson.memory["RAM"]["tot"],
        "CPU Temp": jetson.temperature["CPU"]["temp"],
        "GPU Temp": jetson.temperature["GPU"]["temp"],
        "CPU Power Consumption": jetson.power["rail"]["POM_5V_CPU"]["volt"]
        * jetson.power["rail"]["POM_5V_CPU"]["curr"],  # watts!
        "GPU Power Consumption": jetson.power["POM_5V_GPU"][
            "power"
        ],  # GPU power in mW or W (not entirely sure yet)
    }


def measure(target_pid, training_type):
    """
    Collects metrics from a Jetson device and writes them to a CSV file.

    Args:
        target_pid (int): The process ID of the target process.
        training_type (str): The type of training being performed.

    Returns:
        None
    """
    filename = f"/media/nano/Nano Micro SD/measurements/benchmarks/metrics-{training_type}_{datetime.now().strftime('%Y-%m-%d@%H-%M-%S')}.csv"
    with jtop() as jetson:
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            while jetson.ok():
                metrics = get_metrics(jetson)
                writer.writerow(metrics)
                time.sleep(2.5)

    print(f"Finished collecting metrics - (calling process <{target_pid}> killed)")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Collect metrics from a Jetson device."
    )
    parser.add_argument(
        "--pid", type=int, required=True, help="Process ID of the target process."
    )
    parser.add_argument(
        "--train", type=str, required=True, help="Specify the type of training"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    measure(args.pid, args.train)
