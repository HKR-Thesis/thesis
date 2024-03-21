import sys
from src.q_learning.train_episodes import simulate_episodes as classic_sim
from src.numba_desktop.train_episodes import simulate_episodes as numba_sim

types = ['classic', 'numba', 'dql']

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in types:
        print('Usage: python3.x -m src.train <training_type>')
        print(f"Valid values for <training_type> include {', '.join(types)}")
        sys.exit(1)
    
    try:
        training_type = sys.argv[1]
        if training_type == 'classic':
            q_learning, total_rewards = classic_sim(15000)
        elif training_type == 'numba':
            q_learning, total_rewards = numba_sim(15000)

    except Exception as e:
        print(f"Something went wrong here: {e.with_traceback}")

if __name__ == "__main__":
    main()
