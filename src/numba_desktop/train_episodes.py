from ..inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from .q_learning import QLearning
import numpy as np


def simulate_episodes(num_ep: int):
    inverted_pendulum = InvertedPendulum()

    config = [
        1,  # "alpha": 0
        1,  # "gamma": 1
        0.2,  # "epsilon": 2
        num_ep,  # "number_of_episodes": 3
        # "bins": {
        30,  # "theta": 4
        30,  # "theta_dot": 5
        30,  # "cart_position": 6
        30,  # "cart_velocity": 7
        # "low_bounds": {
        2.71,  # "theta": 8
        6,  # "theta_dot": 9
        0.0,  # "cart_position": 10
        -1.8,  # "cart_velocity": 11
        # "up_bounds": {
        3.58,  # "theta": 12
        -6,  # "theta_dot": 13
        0.5,  # "cart_position": 14
        1.8,  # "cart_velocity": 15
        # "actions": [
        -60,  # action_one: 16
        60,  # action_two: 17
    ]

    total_rewards = []
    q_learning = QLearning(config)
    for episode_index in range(num_ep):
        rewards_episode = []

        inverted_pendulum.reset()

        done = False
        while not done:
            state = np.array(inverted_pendulum.state)
            disc_state = QLearning.discretize_state(
                state,
                q_learning.low_bounds,
                q_learning.up_bounds,
                q_learning.bins,
            )
            action = q_learning.select_action(disc_state, episode_index)  # type: ignore
            _, reward, done = inverted_pendulum.simulate_step(action)  # not optimized
            new_state = np.array(inverted_pendulum.state)
            next_state_disc = QLearning.discretize_state(
                new_state,
                q_learning.low_bounds,
                q_learning.up_bounds,
                q_learning.bins,
            )

            rewards_episode.append(reward)
            q_learning.q_table = QLearning.update_q_table(
                q_learning.q_table,
                disc_state,
                action,
                reward,
                next_state_disc,
                done,
                q_learning.gamma,
                q_learning.alpha,
            )

        print(f"Episode {episode_index} - Total reward: {sum(rewards_episode)}")
        total_rewards.append(sum(rewards_episode))

    return q_learning, total_rewards
