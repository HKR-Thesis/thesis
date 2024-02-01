from .train_episodes import simulate_episodes
from .simulate_learned_strategy import simulate_learned_strategy
import matplotlib.pyplot as plt


def reward_plot(total_rewards: list[float]):
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()


def main():
    q_learning, total_rewards = simulate_episodes(15000)
    reward_plot(total_rewards)
    simulate_learned_strategy(q_learning, 1000)
    print(q_learning.q_table)
    print(total_rewards)


if __name__ == "__main__":
    main()
