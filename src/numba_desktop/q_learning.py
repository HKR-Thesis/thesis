import numpy as np
from ..inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from numba import njit
from numba import float32


class QLearning:
    def __init__(self, config: list) -> None:

        self.alpha = config[0]
        self.gamma = config[1]
        self.epsilon = config[2]
        self.num_ep = config[3]
        self.bins = np.array(
            [config[4], config[5], config[6], config[7]], dtype=np.int32
        )
        self.low_bounds = np.array(
            [config[8], config[9], config[10], config[11]], dtype=np.float64
        )
        self.up_bounds = np.array(
            [config[12], config[13], config[14], config[15]], dtype=np.float64
        )
        self.actions = [config[16], config[17]]

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
    ) -> tuple[np.float64, np.float64, np.float64, np.float64]:
        (
            angle_theta,
            angular_velocity_theta_dot,
            cart_position,
            cart_velocity,
        ) = simulator_state

        theta_bins = np.linspace(
            low_bounds[0],
            up_bounds[0],
            bins[0],
        )
        theta_dot_bins = np.linspace(
            low_bounds[1],
            up_bounds[1],
            bins[1],
        )
        cart_position_bins = np.linspace(
            low_bounds[2],
            up_bounds[2],
            bins[2],
        )
        cart_velocity_bins = np.linspace(
            low_bounds[3],
            up_bounds[3],
            bins[3],
        )

        theta_index = np.digitize(angle_theta, theta_bins)
        theta_dot_index = np.digitize(
            np.abs(angular_velocity_theta_dot), theta_dot_bins
        )
        cart_position_index = np.digitize(cart_position, cart_position_bins)
        cart_velocity_index = np.digitize(cart_velocity, cart_velocity_bins)

        return (
            np.subtract(theta_index, 1),
            np.subtract(theta_dot_index, 1),
            np.subtract(cart_position_index, 1),
            np.subtract(cart_velocity_index, 1),
        )

    def select_action(
        self, state: tuple[np.intp, np.intp, np.intp, np.intp], episode_index: int
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

    def update_q_table(
        self,
        state: tuple[np.intp, np.intp, np.intp, np.intp],
        action: np.intp,
        reward: float,
        next_state: tuple[np.intp, np.intp, np.intp, np.intp],
        terminal_state: bool,
    ) -> None:
        q_max_prime = np.max(self.q_table[next_state])

        if terminal_state:
            error = reward + self.gamma * q_max_prime - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * error
        else:
            error = reward - self.q_table[state + (action,)]
            self.q_table[state][action] += self.alpha * error
