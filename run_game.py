import argparse
import os
import h5py
import gym
import numpy as np
import pandas as pd
import random
from time import sleep
from numpy.random import randint
from collections import deque


from keras.initializers import normal, identity

from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD , Adam, Adagrad, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Merge
from keras.models import Model


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


GAME = 'Boxing-ram-v0'
aux = gym.make(GAME)

BATCH_SIZE = 10
MAX_ITERATIONS_PER_EPISODE = 200
LEARNING_RATE = 0.00025
TARGET_UPDATE_LIMIT = 1
EPSILON = 1.0
EPSILON_UPDATE = 0.1
MIN_EPSILON = 0.1
GAMMA = 0.95
TERMINAL_REWARD = 0
NUMBER_OF_FRAMES = 1

INPUT_SIZE = aux.observation_space.shape[0]
OUTPUT_SIZE = aux.action_space.n

MIN_EXPERIENCE_REPLAY_SIZE = 1000
MAX_EXPERIENCE_REPLAY_SIZE = 100000
DATA_TYPE = np.uint8


def build_model(learning_rate):
    """"Return the neural network"""
    model = Sequential()
    model.add(Dense(1024, kernel_initializer='he_normal', activation='relu', input_dim=(INPUT_SIZE * NUMBER_OF_FRAMES)))
    model.add(BatchNormalization())
    model.add(Dense(1024, kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(OUTPUT_SIZE, kernel_initializer='truncated_normal'))

    optimizer = SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


class DeepQNetwork:

    def __init__(self, **kwargs):
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = BATCH_SIZE

        if 'min_experience_replay_size' in kwargs:
            self.minimum_experience_replay_size = kwargs['min_experience_replay_size']
        else:
            self.minimum_experience_replay_size = MIN_EXPERIENCE_REPLAY_SIZE

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = LEARNING_RATE

        if 'epsilon' in kwargs:
            self.epsilon = kwargs['epsilon']
        else:
            self.epsilon = EPSILON

        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        else:
            self.gamma = GAMMA

        if 'target_update_limit' in kwargs:
            self.target_update_limit = kwargs['target_update_limit']
        else:
            self.target_update_limit = TARGET_UPDATE_LIMIT

        if 'max_iterations_per_episode' in kwargs:
            self.max_iterations_per_episode = kwargs['max_iterations_per_episode']
        else:
            self.max_iterations_per_episode = MAX_ITERATIONS_PER_EPISODE

        if 'min_epsilon' in kwargs:
            self.min_epsilon = kwargs['min_epsilon']
        else:
            self.min_epsilon = MIN_EPSILON

        if 'terminal_reward' in kwargs:
            self.terminal_reward = kwargs['terminal_reward']
        else:
            self.terminal_reward = TERMINAL_REWARD

        if 'epsilon_update' in kwargs:
            self.epsilon_update = kwargs['epsilon_update']
        else:
            self.epsilon_update = EPSILON_UPDATE

        self.last_action = None
        self.last_states = deque()
        self.total_last_states = 0

        self.replay = deque()
        self.total_replay = 0

        self.env = gym.make(GAME)

        self.reward_sum = 0
        self.total_episodes = 0

        self.episode_iterations = 0
        self.target_update = 0

        self.model = build_model(self.learning_rate)
        self.frozen_model = build_model(self.learning_rate)

    def choose_best_action(self):
        """Return the action a that maximizes q(self.last_states, a)"""
        s = np.array([np.concatenate([self.last_states]).reshape(NUMBER_OF_FRAMES * INPUT_SIZE)])
        q = self.model.predict(s / 256.)[0]
        action = np.argmax(q)
        return action, q[action]

    def choose_random_action(self):
        return self.env.action_space.sample()

    def choose_e_greedy_action(self):
        """Return an action chosen following the e-greedy policy"""
        if random.random() <= self.epsilon or self.total_last_states < NUMBER_OF_FRAMES:
            return self.choose_random_action()
        else:
            action, _ = self.choose_best_action()
            return action

    def execute_action(self, action):
        """Return the reward for executing the action and a boolean
           indicating if the new state is terminal.

        """
        state, original_reward, done, _ = self.env.step(action)
        reward = original_reward
        self.reward_sum += reward

        #         if done:
        #             reward = self.terminal_reward

        self.add_replay(action, reward, state, done)

        self.episode_iterations += 1
        if done or self.episode_iterations >= self.max_iterations_per_episode:
            self.episode_iterations = 0
            done = True
            self.total_episodes += 1
            self.reset_states()
        else:
            self.add_state(state)

        return original_reward, done

    def add_replay(self, action, reward, state, done):
        previous_state = None
        if self.total_last_states == NUMBER_OF_FRAMES:
            previous_state = self.copy_last_states()

        self.add_state(state)

        if previous_state is not None:
            current_state = self.copy_last_states()
            self.replay.append((previous_state, action, reward, current_state, done))
            self.total_replay += 1
            if self.total_replay > MAX_EXPERIENCE_REPLAY_SIZE:
                self.replay.popleft()
                self.total_replay -= 1

    def copy_last_states(self):
        v = []
        for s in self.last_states:
            v.append(np.copy(s))
        return v

    def add_state(self, state):
        self.last_states.append(np.array(state, dtype=DATA_TYPE))
        self.total_last_states += 1
        if self.total_last_states > NUMBER_OF_FRAMES:
            self.last_states.popleft()
            self.total_last_states -= 1

    def reset_states(self):
        self.last_states.clear()
        self.last_states.append(np.array(self.env.reset(), dtype=DATA_TYPE))
        self.total_last_states = 1

    def run_test_average_reward(self, total_episodes, render):
        """Run the environment without traning.

           Keyword arguments:
           total_episodes -- number of times the environment
               will be run.
           render -- boolean indicating if the screen must be
               rendered
        """
        reward_sum = 0
        total_iterations = 0

        for _ in range(0, total_episodes):
            self.episode_iterations = 0
            self.reset_states()
            done = False
            reward = 0

            while not done:
                if render:
                    sleep(0.03)
                    self.env.render()

                if self.total_last_states < NUMBER_OF_FRAMES:
                    action = self.choose_random_action()
                else:
                    #                     print self.total_last_states
                    #                     print self.last_states.shape
                    action, _ = self.choose_best_action()
                total_iterations += 1

                r, done = self.execute_action(action)
                reward += r
            reward_sum += reward
            if render:
                sleep(2.0)

        avg_reward = reward_sum / float(total_iterations)

        return avg_reward

    def run_test_average_qvalue(self):
        """Calculate the average max q-value for the random_states"""
        y = self.model.predict(random_states)
        return np.average(np.amax(y, axis=1))

    def update_network(self):
        """Execute a mini-batch update
        """
        batch = random.sample(self.replay, self.batch_size - 1)
        batch.append(self.replay[-1])

        X_last = np.zeros((self.batch_size, NUMBER_OF_FRAMES * INPUT_SIZE), dtype=DATA_TYPE)
        X_current = np.zeros((self.batch_size, NUMBER_OF_FRAMES * INPUT_SIZE), dtype=DATA_TYPE)

        for i in range(0, self.batch_size):
            ls, la, r, s, d = batch[i]
            X_last[i] = np.concatenate([ls]).reshape(NUMBER_OF_FRAMES * INPUT_SIZE)
            X_current[i] = np.concatenate([s]).reshape(NUMBER_OF_FRAMES * INPUT_SIZE)

        y = self.model.predict(X_last / 256.)

        self.target_update += 1
        if self.target_update >= self.target_update_limit:
            self.target_update = 0
            self.frozen_model.set_weights(self.model.get_weights())

        q_theta = self.frozen_model.predict(X_current / 256.)

        for i in range(0, self.batch_size):
            _, la, r, _, d = batch[i]

            if d:
                score = r
            else:
                score = r + self.gamma * np.max(q_theta[i])

            y[i][la] = score

        loss = self.model.train_on_batch(X_last / 256., y)

    #         loss = self.model.fit(X_last, y, batch_size=32, nb_epoch=3, verbose=1)

    def train(self, total_frames, render):
        """Run the neural network training

           Keyword arguments:
           total_frames -- number of times the training process will be executed
           render -- if the screen should be rendered

        """
        self.reset_states()

        training_iterations = 0

        self.episode_iterations = 0
        while (training_iterations < total_frames):
            if render:
                self.env.render()

            action = self.choose_e_greedy_action()
            self.execute_action(action)

            if self.total_replay > MIN_EXPERIENCE_REPLAY_SIZE:
                training_iterations += 1
                self.update_network()

        # update epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_update)


if __name__ == '__main__':
    config = {
        'max_iterations_per_episode': 5000000
    }

    dqn = DeepQNetwork(**config)
    dqn.model.load_weights('models/boxing_3.h5')

    dqn.run_test_average_reward(1000, True)
