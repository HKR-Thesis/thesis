from src.deep_q_learning.train_episodes import simulate_episodes
from src.deep_q_learning.simulate_model import simulate_model
import matplotlib.pyplot as plt
import time
import keras
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
    start_time = time.time()
    total_rewards = simulate_episodes(1000)
    end_time = time.time()
    save_time(start_time, end_time)
    reward_plot(total_rewards)
    loaded_model = keras.models.load_model("trained_model-dqt.h5")
    simulate_model(loaded_model, 1000)
    print(total_rewards)


if __name__ == "__main__":
    main()
