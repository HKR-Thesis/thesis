from src.training.deep_q_learning.deep_q_learning import DeepQLearning
from src.inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
import numpy as np


def training_episodes(
    env: InvertedPendulum, deep_Q_Learning: DeepQLearning, num_ep: int
):
    sum_rewards = []
    for indexEpisode in range(num_ep):
        print(f"Simulating episode {indexEpisode}")

        rewards_ep = []
        current_state = env.reset()
        terminal = False
        while not terminal:
            action = deep_Q_Learning.select_action(current_state, indexEpisode)
            nextState, reward, terminal = env.simulate_step(action)
            rewards_ep.append(reward)
            deep_Q_Learning.replay_buffer.append(
                (current_state, action, reward, nextState, terminal)
            )
            deep_Q_Learning.train_network()
            current_state = nextState

        print(f"Sum of rewards {np.sum(rewards_ep)}")

        sum_rewards.append(np.sum(rewards_ep))

    return sum_rewards


if __name__ == "__main__":
    env = InvertedPendulum()

    config = {
        "env": env,
        "gamma": 1,
        "epsilon": 0.1,
        "num_ep": 1000,
        "state_dimension": 4,
        "action_dimension": 2,
        "replay_buffer_size": 300,
        "batch_replay_buffer_size": 100,
        "update_target_network_period": 100,
        "counter_update_target_network": 0,
    }
    Deep_Q_Learning = DeepQLearning(config)
    training_episodes(env, Deep_Q_Learning, 100)

    Deep_Q_Learning.main_network.summary()
    Deep_Q_Learning.main_network.save("trained_model_temp.h5")
