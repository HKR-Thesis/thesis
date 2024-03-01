import numpy as np
import random
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
import tensorflow.python.keras.optimizers as optimizers
from collections import deque
from tensorflow import gather_nd
from tensorflow.python.keras.losses import mean_squared_error


class DeepQLearning:
    def __init__(self, config: dict) -> None:
        match config:
            case {
                "gamma": gamma,
                "epsilon": epsilon,
                "state_dimension": state_dimension,
                "action_dimension": action_dimension,
                "replay_buffer_size": replay_buffer_size,
                "batch_replay_buffer_size": batch_replay_buffer_size,
                "update_target_network_period": update_target_network_period,
                "counter_update_target_network": counter_update_target_network,
            }:
                self.gamma = gamma
                self.epsilon = epsilon
                self.state_dimension = state_dimension
                self.action_dimension = action_dimension
                self.replay_buffer_size = replay_buffer_size
                self.batch_replay_buffer_size = batch_replay_buffer_size
                self.update_target_network_period = update_target_network_period
                self.counter_update_target_network = counter_update_target_network
            case _:
                raise ValueError("Invalid configuration")

        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        self.main_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.main_network.get_weights())
        self.actions = []

    def loss_fn(self, y_true, y_pred):
        s1, s2 = y_true.shape
        indices = np.zeros(shape=(s1, s2))
        indices[:, 0] = np.arange(s1)
        indices[:, 1] = self.actions
        loss = mean_squared_error(
            gather_nd(y_true, indices=indices.astype(int)),
            gather_nd(y_pred, indices=indices.astype(int)),
        )
        return loss

    def create_network(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_dimension, activation="relu"))
        model.add(Dense(56, activation="relu"))
        model.add(Dense(self.action_dimension, activation="linear"))
        model.compile(
            optimizer="rmsprop",
            loss=self.loss_fn,
            metrics=["accuracy"],
        )
        return model

    def select_action(self, state, index):
        import numpy as np

        if index < 1:
            return np.random.choice(self.action_dimension)
        randomNumber = np.random.random()
        if index > 200:
            self.epsilon = 0.999 * self.epsilon
        if randomNumber < self.epsilon:
            return np.random.choice(self.action_dimension)
        else:
            Qvalues = self.main_network.predict([state])
            return np.random.choice(np.where(Qvalues[0, :] == np.max(Qvalues[0, :]))[0])

    def train_network(self):
        if len(self.replay_buffer) > self.batch_replay_buffer_size:
            randomSampleBatch = random.sample(
                self.replay_buffer, self.batch_replay_buffer_size
            )
            currentStateBatch = np.zeros(shape=(self.batch_replay_buffer_size, 4))
            nextStateBatch = np.zeros(shape=(self.batch_replay_buffer_size, 4))
            for index, tupleS in enumerate(randomSampleBatch):
                currentStateBatch[index, :] = tupleS[0]
                nextStateBatch[index, :] = tupleS[3]
            QnextStateTargetNetwork = self.target_network.predict(nextStateBatch)
            QcurrentStateMainNetwork = self.main_network.predict(currentStateBatch)
            inputNetwork = currentStateBatch
            outputNetwork = np.zeros(shape=(self.batch_replay_buffer_size, 2))
            self.actions = []
            for index, (
                currentState,
                action,
                reward,
                nextState,
                terminated,
            ) in enumerate(randomSampleBatch):
                if terminated:
                    y = reward
                else:
                    y = reward + self.gamma * np.max(QnextStateTargetNetwork[index])
                self.actions.append(action)
                outputNetwork[index] = QcurrentStateMainNetwork[index]
                outputNetwork[index, action] = y
            self.main_network.fit(
                inputNetwork,
                outputNetwork,
                batch_size=self.batch_replay_buffer_size,
                epochs=100,
            )
            self.counter_update_target_network += 1
            if self.counter_update_target_network > (
                self.update_target_network_period - 1
            ):
                self.target_network.set_weights(self.main_network.get_weights())
                print("Target network updated!")
                print("Counter value {}".format(self.counter_update_target_network))
                self.counter_update_target_network = 0
