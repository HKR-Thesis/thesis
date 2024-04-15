import tensorflow as tf
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.losses import mean_squared_error


class DeepQLearning:
    def __init__(self, config: dict):
        required_keys = [
            "gamma",
            "epsilon",
            "state_dimension",
            "action_dimension",
            "buffer_size",
            "batch_size",
            "tn_update_period",
        ]
        if all(key in config for key in required_keys):
            self.gamma = config["gamma"]
            self.epsilon = config["epsilon"]
            self.state_dimension = config["state_dimension"]
            self.action_dimension = config["action_dimension"]
            self.buffer_size = config["buffer_size"]
            self.batch_size = config["batch_size"]
            self.tn_update_period = config["tn_update_period"]
        else:
            raise ValueError("Invalid configuration")

        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.online_model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.online_model.get_weights())
        self.step = 0
        self.actions = np.array([])

    def create_model(self):
        model = Sequential(
            [
                Dense(128, input_dim=self.state_dimension, activation="elu"),
                Dense(64, activation="elu"),
                Dense(self.action_dimension, activation="linear"),
            ]
        )
        model.compile(optimizer="rmsprop", loss=self.loss_fn, metrics=["accuracy"])
        return model

    def loss_fn(self, true, pred):
        indices = tf.cast(self.actions, tf.int32)
        true_selected = tf.gather_nd(true, indices)
        pred_selected = tf.gather_nd(pred, indices)

        loss = mean_squared_error(true_selected, pred_selected)
        return loss

    def select_action(self, state, episode_index):
        if episode_index > 20:
            self.epsilon *= 0.999
        if episode_index < 1:
            return np.random.choice([-1, 1])

        random_number = np.random.random()
        if random_number < self.epsilon:
            return np.random.choice([-1, 1])
        else:
            q_values = self.online_model.predict(np.array([state]), verbose=0)
            return np.argmax(q_values[0])

    def sample_batches(self):
        if len(self.replay_buffer) < self.batch_size:
            raise ValueError("Not enough samples in replay_buffer")

        indices = np.random.choice(
            len(self.replay_buffer), self.batch_size, replace=False
        )
        random_sample_batch = [self.replay_buffer[i] for i in indices]
        current_batch = np.array([transition[0] for transition in random_sample_batch])
        next_batch = np.array([transition[3] for transition in random_sample_batch])

        return random_sample_batch, current_batch, next_batch

    def train_network(self):
        if len(self.replay_buffer) <= self.batch_size:
            return

        random_sample_batch, current_batch, next_batch = self.sample_batches()

        tn_next_state = self.target_model.predict(next_batch, verbose=0)
        on_current_state = self.online_model.predict(current_batch, verbose=0)

        input_network = current_batch
        output_network = np.zeros(shape=(self.batch_size, 2))
        self.actions = np.zeros(shape=(self.batch_size, 1))

        for index, (_, action, reward, _, terminated) in enumerate(random_sample_batch):
            if terminated:
                reward_with_error = reward
            else:
                reward_with_error = reward + self.gamma * np.max(tn_next_state[index])
            self.actions[index] = action

            output_network[index] = on_current_state[index]
            output_network[index, action] = reward_with_error

        self.online_model.fit(
            input_network,
            output_network,
            batch_size=self.batch_size,
            epochs=8,
            verbose=0,
        )
        self.step += 1

        if self.step > (self.tn_update_period - 1):
            self.target_model.set_weights(self.online_model.get_weights())
            self.step = 0
