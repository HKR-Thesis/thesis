import numpy as np
from src.inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from src.inverted_pendulum_simulator.src.inverted_pendulum_visualizer import (
    InvertedPendulumVisualizer,
)
from .q_learning import QLearning


def simulate_learned_strategy(q_learning, episodes: int) -> None:
    env = InvertedPendulum()
    visualizer = InvertedPendulumVisualizer(env)
    states = []

    for i in range(episodes):
        states.append(env.state)
        state = np.array(env.state)
        disc_state = QLearning.discretize_state(
            state,
            q_learning.low_bounds,
            q_learning.up_bounds,
            q_learning.bins,
        )
        action = np.argmax(q_learning.q_table[disc_state])
        env.simulate_step(action)

    visualizer.animate_states(states)
