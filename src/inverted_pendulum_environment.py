from .inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
import numpy as np


class InvertedPendulumEnvironment:
    def __init__(self, number_of_bins: dict, epsilon: float = 0.1):
        self.simulator = InvertedPendulum()
        self.upper_bounds = {
            "theta": 3.58,
            "theta_dot": 6.28,  # Absolute value
            "cart_position": 0.5,
            "cart_velocity": 1.8,
        }

        self.lower_bounds = {
            "theta": 2.71,
            "theta_dot": 0.09,
            "cart_position": 0.0,
            "cart_velocity": -1.8,
        }

        self.actions = {
            "left": -60,
            "right": 60,
        }

        self.q_table = np.random.uniform(
            low=0,
            high=1,
            size=(
                number_of_bins["theta"],
                number_of_bins["theta_dot"],
                number_of_bins["cart_position"],
                number_of_bins["cart_velocity"],
                len(self.actions),
            ),
        )

        self.number_of_bins = number_of_bins
        self.epsilon = epsilon

        # Define action and state spaces

    def discretize_state(self):
        (
            angle_theta,
            angular_velocity_theta_dot,
            cart_position,
            cart_velocity,
        ) = self.simulator.state

        theta_bins = np.linspace(
            self.lower_bounds["theta"],
            self.upper_bounds["theta"],
            self.number_of_bins["theta"],
        )
        theta_dot_bins = np.linspace(
            self.lower_bounds["theta_dot"],
            self.upper_bounds["theta_dot"],
            self.number_of_bins["theta_dot"],
        )
        cart_position_bins = np.linspace(
            self.lower_bounds["cart_position"],
            self.upper_bounds["cart_position"],
            self.number_of_bins["cart_position"],
        )
        cart_velocity_bins = np.linspace(
            self.lower_bounds["cart_velocity"],
            self.upper_bounds["cart_velocity"],
            self.number_of_bins["cart_velocity"],
        )

        theta_index = np.digitize(angle_theta, theta_bins)
        theta_dot_index = np.digitize(
            np.abs(angular_velocity_theta_dot), theta_dot_bins
        )
        cart_position_index = np.digitize(cart_position, cart_position_bins)
        cart_velocity_index = np.digitize(cart_velocity, cart_velocity_bins)

        return theta_index, theta_dot_index, cart_position_index, cart_velocity_index

    def reset(self):
        self.simulator = InvertedPendulum()
        return self.discretize_state()

    def step(self, action):
        # Apply the action to the simulator
        new_state = self.simulator.simulate_step(voltage=action)
        # Calculate the reward based on the new state
        reward = self.calculate_reward(new_state)
        # Check for termination condition
        done = self.is_done(new_state)
        return new_state, reward, done

    def calculate_reward(self, state):
        theta, _, _, _ = state
        target_angle = 3.125

        angle_difference = np.abs(theta - target_angle)

        max_angle = 3.58
        min_angle = 2.71

        reward = 1.0 / (1.0 + angle_difference)

        if theta >= max_angle or theta <= min_angle:
            reward -= 2.0

        return reward

    def select_action(self, episode_index: int) -> str:
        if episode_index < 100:
            return np.random.choice(
                list(
                    self.actions.keys(),
                )
            )

        random_number = np.random.uniform()

        if episode_index > 1000:
            self.epsilon *= 0.99

        if random_number < self.epsilon:
            return np.random.choice(
                list(
                    self.actions.keys(),
                )
            )

        index_state = self.discretize_state()
        action_index = np.argmax(self.q_table[index_state])
        return list(self.actions.keys())[action_index]


if __name__ == "__main__":
    env = InvertedPendulumEnvironment(
        {
            "theta": 10,
            "theta_dot": 10,
            "cart_position": 10,
            "cart_velocity": 10,
        }
    )
    state = env.reset()
    print(state)
