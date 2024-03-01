import matplotlib.pyplot as plt
import time
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
    with open("execution_results_numba.txt", "w") as file:
        file.write(f"Execution Time: {execution_time} seconds\n")


def main():
    start_time = time.time()
    q_learning, total_rewards = simulate_episodes(15000)
    end_time = time.time()
    save_time(start_time, end_time)
    reward_plot(total_rewards)
    simulate_learned_strategy(q_learning, 1000)  # type: ignore
    print(q_learning.q_table)
    print(total_rewards)


if __name__ == "__main__":
    main()
