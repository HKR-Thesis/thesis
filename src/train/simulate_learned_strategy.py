import numpy as np
from src.inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from src.inverted_pendulum_simulator.src.inverted_pendulum_visualizer import (
    InvertedPendulumVisualizer,
)
from src.train.q_learning import QLearning


def simulate_learned_strategy(q_learning: QLearning, episodes: int) -> None:
    env = InvertedPendulum()
    visualizer = InvertedPendulumVisualizer(env)
    states = []

    for i in range(episodes):
        states.append(env.state)
        disc_state = q_learning.discretize_state(env)
        action = np.argmax(q_learning.q_table[disc_state])
        env.simulate_step(action)

    visualizer.animate_states(states)
