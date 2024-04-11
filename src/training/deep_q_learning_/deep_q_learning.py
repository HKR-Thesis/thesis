import numpy as np
import tensorflow as tf
from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.losses import mean_squared_error
from src.training.tensorboard_utils import write_tensorboard_logs


class DeepQLearning:
    def __init__(self, config: dict) -> None:
        match config:
            case {
                "gamma": gamma,
                "epsilon": epsilon,
                "state_dimension": state_dimension,
                "action_dimension": action_dimension,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
            }:
                self.gamma = gamma
                self.epsilon = epsilon
                self.state_dimension = state_dimension
                self.action_dimension = action_dimension
                self.buffer_size = buffer_size
                self.batch_size = batch_size
            case _:
                raise ValueError("Invalid configuration")

        self.writer = tf.summary.create_file_writer(logdir="/out/logs/")
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.step = 0
        self.online_model = self.create_model()
        self.actions = np.array([])

    def create_model(self):
        model = Sequential(
            [
                # Dense(128, input_dim=self.state_dimension, activation="elu"),
                # Dense(64, activation="elu"),
                Dense(
                    self.action_dimension, activation="linear"
                ),  # Solely try a linear approximation
            ]
        )
        model.compile(
            optimizer="rmsprop",
            loss=self.loss_fn,
            metrics=["accuracy"],
        )
        return model

    def loss_fn(self, true, pred):
        indices = tf.cast(self.actions, tf.int32)
        true_selected = tf.gather_nd(true, indices)
        pred_selected = tf.gather_nd(pred, indices)

        loss = mean_squared_error(true_selected, pred_selected)
        return loss

    def select_action(self, state, episode_index):
        if episode_index > 400:
            self.epsilon *= 0.999
        if episode_index < 200:
            return np.random.choice([0, 1])

        random_number = np.random.random()

        if random_number < self.epsilon:
            return np.random.choice([0, 1])
        else:
            q_values = self.online_model.predict([state], verbose=0)  # type: ignore
            return np.argmax(q_values[0])

    def sample_batches(self):
        if len(self.replay_buffer) < self.batch_size:
            raise ValueError("Not enough samples in replay_buffer")

        # Randomly sample indices
        indices = np.random.choice(
            len(self.replay_buffer), self.batch_size, replace=False
        )

        random_sample_batch = [self.replay_buffer[i] for i in indices]
        current_batch = np.array([transition[0] for transition in random_sample_batch])

        return random_sample_batch, current_batch

    def train_network(self):
        with self.writer.as_default():

            if len(self.replay_buffer) <= self.batch_size:
                return

            random_sample_batch, current_batch = self.sample_batches()

            on_curr_state = self.online_model.predict(current_batch, verbose=0)  # type: ignore

            input_network = current_batch
            output_network = np.zeros(shape=(self.batch_size, 2))
            self.actions = np.zeros(shape=(self.batch_size, 1))

            for index, (_, action, reward, _, terminated) in enumerate(
                random_sample_batch
            ):
                if terminated:
                    reward_with_error = reward
                else:
                    reward_with_error = reward + self.gamma * np.max(
                        on_curr_state[index]
                    )
                self.actions[index] = action

                output_network[index] = on_curr_state[index]
                output_network[index, action] = reward_with_error

            self.online_model.fit(
                input_network,
                output_network,
                batch_size=self.batch_size,
                epochs=8,  # Epochs on each batch
                verbose=0,  # type: ignore
            )
            self.step += 1

            write_tensorboard_logs(
                writer=self.writer,
                model=self.online_model,
                step=self.step,
                reward=reward_with_error,
            )
