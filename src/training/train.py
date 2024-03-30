import sys
from src.training.classic.train_episodes import simulate_episodes as classic_sim
from src.training.numba_desktop.train_episodes import simulate_episodes as numba_sim
from src.training.deep_q_learning.train_episodes import simulate_episodes as dql_sim
from src.training.deep_q_learning_.train_episodes import (
    simulate_episodes as dql_target_sim,
)

types = ["classic", "numba", "dql-target", "dql"]


def train():
    if len(sys.argv) < 2 or sys.argv[1] not in types:
        print("Usage: python3.x -m src.training.train <training_type>")
        print(f"Valid values for <training_type> include {', '.join(types)}")
        sys.exit(1)
    try:
        training_type = sys.argv[1]
        print(f"Attempting training for {training_type}")
        if training_type == "classic":
            _, _ = classic_sim(15000)
        elif training_type == "numba":
            _, _ = numba_sim(15000)
        elif training_type == "dql-target":
            _, _ = dql_target_sim(15000)
        elif training_type == "dql":
            _, _ = dql_sim(15000)
    except Exception as e:
        print(f"Something went wrong here: {e.with_traceback}")


if __name__ == "__main__":
    train()
