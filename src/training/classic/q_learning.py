import numpy as np
from src.inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum


class QLearning:
    def __init__(self, config: dict) -> None:
        match config:
            case {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "bins": bins,
                "low_bounds": low_bounds,
                "up_bounds": up_bounds,
                "actions": actions,
            }:
                self.alpha = alpha
                self.gamma = gamma
                self.epsilon = epsilon
                self.bins = bins
                self.low_bounds = low_bounds
                self.up_bounds = up_bounds
                self.actions = actions
            case _:
                raise ValueError("Invalid configuration")

        self.q_table = np.random.uniform(
            low=0,
            high=1,
            size=(
                bins["theta"],
                bins["theta_dot"],
                bins["cart_position"],
                bins["cart_velocity"],
                len(actions),
            ),
        )

    def discretize_state(
        self, simulator: InvertedPendulum
    ) -> tuple[np.intp, np.intp, np.intp, np.intp]:
        (
            angle_theta,
            angular_velocity_theta_dot,
            cart_position,
            cart_velocity,
        ) = simulator.state

        theta_bins = np.linspace(
            self.low_bounds["theta"],
            self.up_bounds["theta"],
            self.bins["theta"],
        )
        theta_dot_bins = np.linspace(
            self.low_bounds["theta_dot"],
            self.up_bounds["theta_dot"],
            self.bins["theta_dot"],
        )
        cart_position_bins = np.linspace(
            self.low_bounds["cart_position"],
            self.up_bounds["cart_position"],
            self.bins["cart_position"],
        )
        cart_velocity_bins = np.linspace(
            self.low_bounds["cart_velocity"],
            self.up_bounds["cart_velocity"],
            self.bins["cart_velocity"],
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
