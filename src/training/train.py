import sys
from src.training.classic.train_episodes import simulate_episodes as classic_sim
from src.training.numba_desktop.train_episodes import simulate_episodes as numba_sim

types = ['classic', 'numba', 'dql']

def train():
    if len(sys.argv) < 2 or sys.argv[1] not in types:
        print('Usage: python3.x -m src.main <training_type>')
        print(f"Valid values for <training_type> include {', '.join(types)}")
        sys.exit(1)
    try:
        training_type = sys.argv[1]
        if training_type == 'classic':
            _, _ = classic_sim(15000)
        elif training_type == 'numba':
            _, _ = numba_sim(15000)
        elif training_type == 'dql':
            print('Not yet implemented. Exiting ..')
    except Exception as e:
        print(f"Something went wrong here: {e.with_traceback}")

train()
