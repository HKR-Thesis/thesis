from src.deep_q_learning_.train_episodes import simulate_episodes
from src.deep_q_learning_.simulate_model import simulate_model
from src.deep_q_learning_.deep_q_learning import DeepQLearning
import matplotlib.pyplot as plt
import time
import keras
import numpy as np
import random
import tensorflow as tf
from datetime import datetime


def reward_plot(total_rewards: list[float]):
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig("figures/" + datetime.today().strftime("%Y-%m-%d-%hr-%m-%s") + ".png")


def save_time(start_time, end_time):

    # Calculate the execution time
    execution_time = end_time - start_time

    # Save the results to a text file
    with open("execution_results_dqn_t.txt", "a") as file:
        file.write(f"Execution Time: {execution_time} seconds\n")


def main():
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    start_time = time.time()
    total_rewards = simulate_episodes(100)
    end_time = time.time()
    save_time(start_time, end_time)
    reward_plot(total_rewards)
    loaded_model = keras.models.load_model(
        "trained_model-dqt.h5", custom_objects={"loss_fn": DeepQLearning.loss_fn}
    )
    simulate_model(loaded_model, 500)
    print(total_rewards)


if __name__ == "__main__":
    main()
