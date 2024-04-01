import sys
import random
import numpy as np
import tensorflow as tf
from src.util import reward_plot
from src.training.classic.train_episodes import simulate_episodes as classic_sim
from src.training.numba_desktop.train_episodes import simulate_episodes as numba_sim
from src.training.deep_q_learning.train_episodes import (
    simulate_episodes as dql_target_sim,
)
from src.training.deep_q_learning_.train_episodes import (
    simulate_episodes as dql_sim,
)

types = ["classic", "numba", "dql-target", "dql"]


def train():
    if len(sys.argv) < 2 or sys.argv[1] not in types:
        print("Usage: python3.x -m src.training.main <training_type>")
        print(f"Valid values for <training_type> include {', '.join(types)}")
        sys.exit(1)
    try:
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        training_type = sys.argv[1]
        print(f"Attempting training for {training_type}")
        rewards = None
        if training_type == "classic":
            _, rewards = classic_sim(15000)
        elif training_type == "numba":
            _, rewards = numba_sim(15000)
        elif training_type == "dql-target":
            rewards = dql_target_sim(100)
        elif training_type == "dql":
            rewards = dql_sim(1000)

        if rewards:
            reward_plot(rewards)
    except Exception as e:
        print(f"Something went wrong here: {e.with_traceback}")


if __name__ == "__main__":
    train()
