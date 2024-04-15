import numpy as np
from numba import njit
from typing import Tuple, List


class QLearning:
    def __init__(self, config: List[float]):
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
    @njit(fastmath=True)
    def discretize_state(
        simulator_state: np.ndarray,
        low_bounds: np.ndarray,
        up_bounds: np.ndarray,
        bins: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        (
            angle_theta,
            angular_velocity_theta_dot,
            cart_position,
            cart_velocity,
        ) = simulator_state

        theta_bins = np.linspace(low_bounds[0], up_bounds[0], bins[0])
        theta_dot_bins = np.linspace(low_bounds[1], up_bounds[1], bins[1])
        cart_position_bins = np.linspace(low_bounds[2], up_bounds[2], bins[2])
        cart_velocity_bins = np.linspace(low_bounds[3], up_bounds[3], bins[3])

        theta_index = np.digitize(angle_theta, theta_bins)
        theta_dot_index = np.digitize(
            np.abs(angular_velocity_theta_dot), theta_dot_bins
        )
        cart_position_index = np.digitize(cart_position, cart_position_bins)
        cart_velocity_index = np.digitize(cart_velocity, cart_velocity_bins)

        return (
            theta_index - 1,
            theta_dot_index - 1,
            cart_position_index - 1,
            cart_velocity_index - 1,
        )

    def select_action(
        self, state: Tuple[int, int, int, int], episode_index: int
    ) -> int:
        if episode_index > 7000:
            self.epsilon *= 0.999
        if episode_index < 5000:
            return np.random.choice(self.actions)

        random_number = np.random.random()

        if random_number < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])

    @staticmethod
    @njit(fastmath=True)
    def update_q_table(
        q_table,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
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
