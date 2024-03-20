import csv, time, sys, os
from datetime import datetime
from jetson_stats import Jtop

def process_is_running(target_pid) -> bool:
    try:
        os.kill(target_pid)
        return True
    except OSError:
        return False

def get_metrics(jetson) -> dict:
    return {
        'Time': datetime.now().strftime('%H:%M:%S'),
        'CPU': jetson.cpu['CPU1'],
        'GPU': jetson.gpu,
        'MEM': jetson.ram['use'] / jetson.ram['tot'],
        'CPU Temp': jetson.temperature['CPU'],
        'GPU Temp': jetson.temperature['GPU'],
        'Power Consumption': jetson.power[1]['5V']
    }


def measure(target_pid):
    filename = f"/out/metrics_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    with Jtop() as jetson:
        with open(filename, mode='w', newline='') as file:
            fieldnames = ['Time', 'CPU', 'GPU', 'MEM', 'CPU Temp', 'GPU Temp', 'Power Consumption']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            while process_is_running(target_pid):
                metrics = get_metrics(jetson)
                writer.writerow(metrics)
                time.sleep(2)
                
    print('Finished collecting metrics - (target process killed)')

if __name__ == '__main__':
    target_pid = int(sys.argv[1])
    measure(target_pid)