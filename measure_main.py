import psutil
import csv
import time


def monitor_process(process_id, output_file):
    with open(output_file, "w", newline="") as csvfile:
        fieldnames = ["CPU (%)", "Memory (%)", "Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        time_ = 0.0
        writer.writeheader()

        while psutil.pid_exists(process_id):
            process = psutil.Process(process_id)
            cpu_percent = process.cpu_percent(interval=1)
            memory_percent = process.memory_percent()
            writer.writerow(
                {
                    "CPU (%)": cpu_percent,
                    "Memory (%)": memory_percent,
                    "Time": time_,
                }
            )
            time_ += 5
            time.sleep(5)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 monitor_process.py <process_id> <output_file>")
        sys.exit(1)

    process_id = int(sys.argv[1])
    output_file = f"{sys.argv[2]}_{process_id}.csv"

    monitor_process(process_id, output_file)
