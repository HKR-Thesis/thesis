"""
Baseline algorithm for reference


Move the cart in the direction of the pendulum angle
Goal is to keep the pendulum upright (theta = 0)
"""

from src.inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from src.inverted_pendulum_simulator.src.inverted_pendulum_visualizer import (
    InvertedPendulumVisualizer,
)

import numpy as np
import matplotlib as plt


def run_simulation(
    pendulum_simulation: InvertedPendulum, states: list[list[float]]
) -> None:
    visualizer = InvertedPendulumVisualizer(pendulum_simulation)
    visualizer.animate_states(states)


def angle_based() -> None:
    pendulum_simulation = InvertedPendulum()
    voltage = 16

    try:
        num_frames = 144 * 100
        states = []

        for _ in range(num_frames):
            angle = pendulum_simulation.state[0]

            if angle > 3.125:
                voltage_command = voltage
            elif angle < 3.125:
                voltage_command = -voltage

            pendulum_simulation.simulate_step(voltage_command)  # type: ignore
            states.append(pendulum_simulation.state.copy())
            print(pendulum_simulation.state)

        states.append(pendulum_simulation.state.copy())
        run_simulation(pendulum_simulation, states)

    except KeyboardInterrupt:
        plt.close()  # type: ignore


def random_based():
    pendulum_simulation = InvertedPendulum()
    voltage = 16

    try:
        num_frames = 144 * 100
        states = []

        for _ in range(num_frames):
            voltage_command = np.random.choice([-voltage, voltage])
            pendulum_simulation.simulate_step(voltage_command)
            states.append(pendulum_simulation.state.copy())
            print(pendulum_simulation.state)

        states.append(pendulum_simulation.state.copy())
        run_simulation(pendulum_simulation, states)

    except KeyboardInterrupt:
        plt.close()  # type: ignore


if __name__ == "__main__":
    # angle_based()
    random_based()
