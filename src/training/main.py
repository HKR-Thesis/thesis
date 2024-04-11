import sys
import random
import numpy as np
import tensorflow as tf
import argparse
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run different types of training simulations."
    )
    parser.add_argument(
        "--train",
        choices=types,
        required=True,
        help="Specify the type of training to perform.",
    )
    return parser.parse_args()


def train(training_type):
    try:
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
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
        print(f"Something went wrong here: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    train(args.train)
