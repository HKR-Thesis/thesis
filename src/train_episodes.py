import numpy as np
from .inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from .q_learning import QLearning


def simulate_episodes(num_ep: int):
    inverted_pendulum = InvertedPendulum()

    config = {
        "alpha": 0.1,
        "gamma": 1,
        "epsilon": 0.2,
        "number_of_episodes": num_ep,
        "bins": {
            "theta": 30,
            "theta_dot": 30,
            "cart_position": 30,
            "cart_velocity": 30,
        },
        "low_bounds": {
            "theta": 2.71,
            "theta_dot": 0.09,
            "cart_position": 0.0,
            "cart_velocity": -1.8,
        },
        "up_bounds": {
            "theta": 3.58,
            "theta_dot": 6.28,
            "cart_position": 0.5,
            "cart_velocity": 1.8,
        },
        "actions": [
            -60,
            60,
        ],
    }

    total_rewards = []
    q_learning = QLearning(config)
    for episode_index in range(num_ep):
        rewards_episode = []

        inverted_pendulum.reset()

        done = False
        while not done:
            disc_state = q_learning.discretize_state(inverted_pendulum)
            action = q_learning.select_action(disc_state, episode_index)
            _, reward, done = inverted_pendulum.simulate_step(action)
            next_state_disc = q_learning.discretize_state(inverted_pendulum)

            rewards_episode.append(reward)
            q_learning.update_q_table(disc_state, action, reward, next_state_disc, done)  # type: ignore

        print(f"Episode {episode_index} - Total reward: {sum(rewards_episode)}")
        total_rewards.append(sum(rewards_episode))

    return q_learning, total_rewards
