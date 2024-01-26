"""
Baseline algorithm for reference


Move the cart in the direction of the pendulum angle
Goal is to keep the pendulum upright (theta = 0)
"""

from .inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from .inverted_pendulum_simulator.src.inverted_pendulum_visualizer import (
    InvertedPendulumVisualizer,
)

import matplotlib as plt


def angle_based() -> None:
    pendulum_simulation = InvertedPendulum()
    voltage = 60

    try:
        num_frames = 144 * 100
        states = []

        for _ in range(num_frames):
            angle = pendulum_simulation.state[0]

            if angle > 3.125:
                voltage_command = voltage
            elif angle < 3.125:
                voltage_command = -voltage
            else:
                voltage_command = 0

            pendulum_simulation.simulate_step(voltage_command)
            states.append(pendulum_simulation.state.copy())
            print(pendulum_simulation.state)

        pendulum_simulation.simulate_step(0)
        states.append(pendulum_simulation.state.copy())
        visualizer = InvertedPendulumVisualizer(pendulum_simulation)
        visualizer.animate_states(states)

    except KeyboardInterrupt:
        plt.close()


if __name__ == "__main__":
    angle_based()
