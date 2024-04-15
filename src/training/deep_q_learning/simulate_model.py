import keras
from src.training.deep_q_learning.deep_q_learning import DeepQLearning
from src.inverted_pendulum_simulator.src.inverted_pendulum import InvertedPendulum
from src.inverted_pendulum_simulator.src.inverted_pendulum_visualizer import (
    InvertedPendulumVisualizer,
)
import numpy as np


def simulate_model(loaded_model, episodes: int):
    env = InvertedPendulum()
    env.reset()
    visualizer = InvertedPendulumVisualizer(env)
    states = []

    for _ in range(episodes):
        states.append(env.state)
        q_values = loaded_model.predict([env.state])
        action = np.argmax(q_values[0])
        env.simulate_step(action)

    visualizer.animate_states(states)


if __name__ == "__main__":
    loaded_model = keras.models.load_model(
        "trained_model-dqt.h5", custom_objects={"loss_fn": DeepQLearning.loss_fn}
    )
    simulate_model(loaded_model, 500)
