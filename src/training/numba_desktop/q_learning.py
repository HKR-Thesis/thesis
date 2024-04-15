import numpy as np
from numba import njit
from typing import Tuple, List


class QLearning:
    def __init__(self, config: List) -> None:

        self.alpha = config[0]
        self.gamma = config[1]
        self.epsilon = config[2]
        self.bins = np.array(
            [config[3], config[4], config[5], config[6]], dtype=np.int32
        )
        self.low_bounds = np.array(
            [config[7], config[8], config[9], config[10]], dtype=np.float64
        )
        self.up_bounds = np.array(
            [config[11], config[12], config[13], config[14]], dtype=np.float64
        )
        self.actions = [config[15], config[16]]

        self.q_table = np.random.uniform(
            low=0,
            high=1,
            size=(
                self.bins[0],
                self.bins[1],
                self.bins[2],
                self.bins[3],
                len(self.actions),
            ),
        )

    @staticmethod
    @njit
    def custom_digitize(value, bins):
        for i in range(len(bins)):
            if value < bins[i]:
                return i
        return len(bins)

    @staticmethod
    @njit
    def discretize_state(simulator_state, low_bounds, up_bounds, bins):
        angle_theta, angular_velocity_theta_dot, cart_position, cart_velocity = (
            simulator_state
        )

        theta_bins = np.linspace(low_bounds[0], up_bounds[0], bins[0] + 1)
        theta_dot_bins = np.linspace(low_bounds[1], up_bounds[1], bins[1] + 1)
        cart_position_bins = np.linspace(low_bounds[2], up_bounds[2], bins[2] + 1)
        cart_velocity_bins = np.linspace(low_bounds[3], up_bounds[3], bins[3] + 1)

        theta_index = QLearning.custom_digitize(angle_theta, theta_bins) - 1
        theta_dot_index = (
            QLearning.custom_digitize(
                np.abs(angular_velocity_theta_dot), theta_dot_bins
            )
            - 1
        )
        cart_position_index = (
            QLearning.custom_digitize(cart_position, cart_position_bins) - 1
        )
        cart_velocity_index = (
            QLearning.custom_digitize(cart_velocity, cart_velocity_bins) - 1
        )

        return (theta_index, theta_dot_index, cart_position_index, cart_velocity_index)

    def select_action(
        self, state: Tuple[np.intp, np.intp, np.intp, np.intp], episode_index: int
    ) -> np.intp:
        if episode_index > 7000:
            self.epsilon *= 0.999
        if episode_index < 5000:
            return np.random.choice([0, 1])

        random_number = np.random.random()

        if random_number < self.epsilon:
            return np.random.choice([0, 1])
        else:
            return np.argmax(self.q_table[state])

    @staticmethod
    @njit(fastmath=True)
    def update_q_table(
        q_table,
        state: Tuple[np.float64, np.float64, np.float64, np.float64],
        action: np.intp,
        reward: float,
        next_state: Tuple[np.float64, np.float64, np.float64, np.float64],
        terminal_state: bool,
        gamma: float,
        alpha: float,
    ) -> np.ndarray:
        q_max_prime = np.max(q_table[next_state])

        if terminal_state:
            error = reward + gamma * q_max_prime - q_table[state][action]
            q_table[state][action] += alpha * error
        else:
            error = reward - q_table[state + (action,)]
            q_table[state][action] += alpha * error

        return q_table
