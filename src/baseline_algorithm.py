"""
Baseline algorithm for reference


Move the cart in the direction of the pendulum angle
Goal is to keep the pendulum upright (theta = 0)
"""

import numpy as np
from src.inverted_pendulum_environment import InvertedPendulumEnvironment
from src.inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from src.inverted_pendulum_simulator.src.inverted_pendulum_visualizer import (
    InvertedPendulumVisualizer,
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time


def main():
    while True:
        pendulum_simulation = InvertedPendulum()
    pass


if __name__ == "__main__":
    main()
