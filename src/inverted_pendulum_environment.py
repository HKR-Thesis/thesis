from .inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
import numpy as np


class InvertedPendulumEnvironment:
    def __init__(self):
        self.simulator = InvertedPendulum()
        # Define action and state spaces

    def reset(self):
        self.simulator = InvertedPendulum()
        # Reset the simulator

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

    def is_done(self, state):
        # Implement the termination condition
        pass


if __name__ == "__main__":
    env = InvertedPendulumEnvironment()
    state = env.reset()
    done = False
    while not done:
        action = 0  # TODO: Implement a controller here
        state, reward, done = env.step(action)
