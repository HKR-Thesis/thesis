import numpy as np
from src.inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from typing import Tuple, Dict, Any

class QLearning:
    def __init__(self, config: Dict[str, Any]) -> None:
        required_keys = ["alpha", "gamma", "epsilon", "bins", "low_bounds", "up_bounds", "actions"]
        if all(key in config for key in required_keys):
            self.alpha = config["alpha"]
            self.gamma = config["gamma"]
            self.epsilon = config["epsilon"]
            self.bins = config["bins"]
            self.low_bounds = config["low_bounds"]
            self.up_bounds = config["up_bounds"]
            self.actions = config["actions"]
        else:
            raise ValueError("Invalid configuration")

        self.q_table = np.random.uniform(
            low=0,
            high=1,
            size=(
                self.bins["theta"],
                self.bins["theta_dot"],
                self.bins["cart_position"],
                self.bins["cart_velocity"],
                len(self.actions),
            ),
        )

    def discretize_state(
        self, simulator: InvertedPendulum
    ) -> Tuple[int, int, int, int]:
        angle_theta, angular_velocity_theta_dot, cart_position, cart_velocity = simulator.state

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
        theta_dot_index = np.digitize(np.abs(angular_velocity_theta_dot), theta_dot_bins)
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

    def update_q_table(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
        terminal_state: bool,
    ) -> None:
        q_max_prime = np.max(self.q_table[next_state])
        if terminal_state:
            error = reward + self.gamma * q_max_prime - self.q_table[state + (action,)]
        else:
            error = reward - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.alpha * error
