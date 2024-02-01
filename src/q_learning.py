import numpy as np
from .inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum


class QLearning:
    def __init__(self, config: dict) -> None:
        match config:
            case {
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "number_of_episodes": num_ep,
                "bins": bins,
                "low_bounds": low_bounds,
                "up_bounds": up_bounds,
                "actions": actions,
            }:
                self.alpha = alpha
                self.gamma = gamma
                self.epsilon = epsilon
                self.num_ep = num_ep
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

        return theta_index, theta_dot_index, cart_position_index, cart_velocity_index

    def select_action(
        self, state: tuple[int, int, int, int], episode_index: int
    ) -> np.intp:
        if episode_index > 1000:
            self.epsilon *= 0.999
        if episode_index < 500:
            return np.random.choice(self.actions)

        random_number = np.random.uniform(0, 1)

        if random_number < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])
