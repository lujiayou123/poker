"""Player based on a trained neural network"""
# pylint: disable=wrong-import-order
import logging
import time

import numpy as np

from gym_env.env import Action

# import torch
# import torch.optim as optim
# from torch import nn as nn
# from torch.optim import Adam

import tensorflow as tf
import json

from keras import Sequential
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout

from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents import DQNAgent
from rl.core import Processor

autplay = True  # play automatically if played against keras-rl

window_length = 1
nb_max_start_steps = 1  # random action
train_interval = 100  # train every 100 steps
nb_steps_warmup = 50  # before training starts, should be higher than start steps
nb_steps = 1000
memory_limit = int(nb_steps / 2)
batch_size = 500  # items sampled from memory to train
enable_double_dqn = False

log = logging.getLogger(__name__)


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', stack_size=2000, range=0.9, env=None , load_model=None):
        """Initiaization of an agent"""
        self.stack = stack_size
        self.range = range
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.dqn = None
        self.model = None
        self.env = env

    def initiate_agent(self, env , load_model=None):
        """initiate a deep Q agent"""
        tf.compat.v1.disable_eager_execution()
        self.env = env

        # print("STATE SPACE:{}".format(env.observation_space))
        #         # print("ACTION SPACE:{}".format(env.action_space[0]))
        nb_actions = self.env.action_space.n
        if load_model:
            self.load(load_model)
        else:
            self.model = Sequential()
            self.model.add(Dense(512, activation='relu', input_shape=env.observation_space))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(nb_actions, activation='linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        policy = TrumpPolicy()
        test_policy = TrumpPolicy()

        # nb_actions = env.action_space.n
        # print("@@@@@@@@@env.action_space:{}".format(nb_actions))

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,test_policy=test_policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        self.dqn.compile(tf.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])

    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm up."""
        log.info("Random action")
        _ = observation
        action = self.env.action_space.sample()
        return action

    def train(self, ckpt_name):
        """Train a model"""
        # initiate training loop
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(ckpt_name)
        tensorboard = TensorBoard(log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True,
                                  write_images=False)

        self.dqn.fit(self.env, nb_max_start_steps=nb_max_start_steps, nb_steps=nb_steps, visualize=False, verbose=2,
                     start_step_policy=self.start_step_policy, callbacks=[tensorboard])

        # Save the architecture
        dqn_json = self.model.to_json()
        with open("dqn_in_json.json", "w") as json_file:
            json.dump(dqn_json, json_file)

        # After training is done, we save the final weights.
        self.dqn.save_weights('dqn_{}_weights.h5'.format(ckpt_name), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        self.dqn.test(self.env, nb_episodes=5, visualize=False)

    def load(self, env_name):
        """Load a model"""

        # Load the architecture
        with open('dqn_in_json.json', 'r') as architecture_json:
            dqn_json = json.load(architecture_json)

        self.model = model_from_json(dqn_json)
        self.model.load_weights('dqn_{}_weights.h5'.format(env_name))

    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)
        policy = TrumpPolicy()
        test_policy = TrumpPolicy()

        class CustomProcessor(Processor):
            """The agent and the environment"""

            def process_state_batch(self, batch):
                """
                Given a state batch, I want to remove the second dimension, because it's
                useless and prevents me from feeding the tensor into my CNN
                """
                return np.squeeze(batch, axis=1)

            def process_info(self, info):
                processed_info = info['player_data']
                if 'stack' in processed_info:
                    processed_info = {'x': 1}
                return processed_info

        nb_actions = self.env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=nb_steps_warmup,
                            target_model_update=1e-2, policy=policy,test_policy=test_policy,
                            processor=CustomProcessor(),
                            batch_size=batch_size, train_interval=train_interval, enable_double_dqn=enable_double_dqn)
        self.dqn.compile(tf.keras.optimizers.Adam(lr=1e-3), metrics=['mae'])  # pylint: disable=no-member
        # self.dqn.compile(tf.train.AdamOptimizer(learning_rate=1e-3), metrics=['mae'])  # pylint: disable=no-member

        self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=render)

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL,Action.ALL_IN,Action.RAISE_200, Action.RAISE_250,Action.RAISE_300,
                                    Action.RAISE_350,Action.RAISE_400,Action.RAISE_450,Action.RAISE_500,Action.RAISE_550,
                                    Action.RAISE_600,Action.RAISE_10_POT,Action.RAISE_20_POT,Action.RAISE_30_POT,Action.RAISE_40_POT,
                                    Action.RAISE_50_POT,Action.RAISE_60_POT,Action.RAISE_70_POT,Action.RAISE_80_POT,Action.RAISE_90_POT,
                                    Action.RAISE_100_POT,Action.RAISE_125_POT,Action.RAISE_150_POT,Action.RAISE_175_POT,Action.RAISE_200_POT}
        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action


class TrumpPolicy(BoltzmannQPolicy):
    """Custom policy when making decision based on neural network."""

    def select_action(self, q_values , legal_moves):#输入所有动作的Q值，根据这些数据选择动作,legal moves表示有效动作空间
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        # print(type(legal_moves))

        # print("SELECT ACTION LEGAL MOVES:{}".format(legal_moves))
        # print(legal_moves[0].value)#ACTION:CALL
        # print(legal_moves[1])
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')#27维,针对所有的action，需要从中挑选出针对legal action的q值
        # print(type(q_values))
        # print("q_values:{}".format(q_values))

        # print(legal_moves)
        legal_moves_indexs = []
        for i in range(len(legal_moves)):
            # print(legal_moves[i].value)
            legal_moves_indexs.append(legal_moves[i].value)
        legal_moves_indexs.sort()

        # print(legal_moves_indexs)#所有合法动作的index


        # nb_legal_actions = len(legal_moves)#27个动作
        # print(nb_legal_actions)
        # print(len(legal_moves))

        nb_actions = q_values.shape[0]#27个动作
        # print("nb_actions:{}".format(nb_actions))

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))#27个动作对应的期望值 nd array
        # print(exp_values)
        legal_exp_values = np.zeros(27)
        # print(legal_exp_values)
        for j in range(len(legal_moves_indexs)):#把不合法动作的期望值全部转化为0
            legal_exp_values[legal_moves_indexs[j]]=exp_values[legal_moves_indexs[j]]
        # print(legal_exp_values)

        # print("exp_values:{}".format(exp_values))
        # print(exp_values[0])

        # probs = exp_values / np.sum(exp_values)#27维，27个概率，加起来等于1
        probs = legal_exp_values / np.sum(legal_exp_values)#27维，27个概率，加起来等于1
        # print(type(probs))
        # print("probs:{}".format(probs))
        # print("sum:{}".format(np.sum(probs)))

        action = np.random.choice(range(nb_actions), p=probs)#range有问题，只能是有效动作                                  把非法动作的期望值在选择动作之前变为0,这样这些动作对应的probs也等于0,不影响sample
        # print("action:{}".format(action))

        log.info(f"Chosen action by keras-rl {action} - probabilities: {probs}")
        return action


class CustomProcessor(Processor):
    """The agent and the environment"""

    def process_state_batch(self, batch):
        """Remove second dimension to make it possible to pass it into cnn"""
        return np.squeeze(batch, axis=1)

    def process_info(self, info):
        if 'legal_moves' in info.keys():
            self.legal_moves_limit = info['legal_moves']
        else:
            self.legal_moves_limit = None
        return {'x': 1}  # on arrays allowed it seems

    def process_action(self, action):
        """Find nearest legal action"""
        # if 'legal_moves_limit' in self.__dict__:
        #     self.legal_moves_limit = [move.value for move in self.legal_moves_limit]
        #     if action not in self.legal_moves_limit:
        #         for i in range(5):
        #             action += i
        #             if action in self.legal_moves_limit:
        #                 break
        #             action -= i * 2
        #             if action in self.legal_moves_limit:
        #                 break
        #             action += i

        return action
