"""
neuron poker

Usage:
  main.py random [options]
  main.py keypress [options]
  main.py consider_equity [options]
  main.py equity_improvement --improvement_rounds=<> [options]
  main.py dqn_train [options]
  main.py dqn_play [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play

"""

import logging

import numpy as np
import pandas as pd
from docopt import docopt

import gym
from agents.agent_consider_equity import Player as EquityPlayer
from agents.agent_keras_rl_dqn import Player as DQNPlayer
from agents.agent_keypress import Player as KeyPressAgent
from agents.agent_random import Player as RandomPlayer
from agents.agent_custom_q1 import Player as Custom_Q1
from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


def command_line_parser():
    """Entry function"""
    args = docopt(__doc__)
    if args['--log']:
        logfile = args['--log']
    else:
        print("Using default log file")
        logfile = 'default'
    screenloglevel = logging.INFO if not args['--screenloglevel'] else \
        getattr(logging, args['--screenloglevel'].upper())
    _ = get_config()
    init_logger(screenlevel=screenloglevel, filename=logfile)
    print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    log.info("Initializing program")

    num_episodes = 1 if not args['--episodes'] else int(args['--episodes'])
    runner = Runner(render=args['--render'], num_episodes=num_episodes, use_cpp_montecarlo=args['--use_cpp_montecarlo'],
                    funds_plot=args['--funds_plot'])

    if args['random']:
        runner.random_agents()

    elif args['keypress']:
        runner.key_press_agents()

    elif args['consider_equity']:
        runner.equity_vs_random()

    elif args['equity_improvement']:
        improvement_rounds = int(args['--improvement_rounds'])
        runner.equity_self_improvement(improvement_rounds)

    elif args['dqn_train']:
        runner.dqn_train_keras_rl()

    elif args['dqn_play']:
        runner.dqn_play_keras_rl()

    else:
        raise RuntimeError("Argument not yet implemented")


class Runner:
    """Orchestration"""

    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot):
        """Initialize"""
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.env = None
        self.num_episodes = num_episodes
        self.log = logging.getLogger(__name__)

    def random_agents(self):
        """Create an environment with 6 random players"""
        env_name = 'neuron_poker-v0'
        stack = 200
        num_of_plrs = 2
        self.env = gym.make(env_name, initial_stacks=stack, render=self.render)
        for _ in range(num_of_plrs):
            player = RandomPlayer()
            self.env.add_player(player)

        self.env.reset()

    def key_press_agents(self):
        """Create an environment with 6 key press agents"""
        env_name = 'neuron_poker-v0'
        stack = 2000
        # num_of_plrs = 6
        env = gym.make(env_name, initial_stacks=stack, render=self.render)
        player = KeyPressAgent(name="LJY",range=0.3)
        env.add_player(player)
        # self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        env.add_player(RandomPlayer(name='Random-1',range=1))
        # env.add_player(RandomPlayer(name='Random-2',range=1))
        # env.add_player(RandomPlayer(name='Random-3',range=1))
        # env.add_player(RandomPlayer(name='Random-4',range=1))
        # env.add_player(RandomPlayer(name='Random-5',range=1))
        # self.env.add_player(PlayerShell(name='dqn001', stack_size=stack))
        # self.env.add_player(PlayerShell(name='dqn002', stack_size=stack))
        # self.env.add_player(PlayerShell(name='dqn003', stack_size=stack))
        # self.env.add_player(PlayerShell(name='dqn004', stack_size=stack))
        # self.env.add_player(PlayerShell(name='dqn005', stack_size=stack))
        env.reset()

        # dqn = DQNPlayer(load_model='dqn1', env=self.env)
        # dqn.play(nb_episodes=self.num_episodes, render=self.render)
        # for _ in range(num_of_plrs):
        #     player = KeyPressAgent()
        #     self.env.add_player(player)

        # self.env.reset()

    def equity_vs_random(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        env_name = 'neuron_poker-v0'
        stack = 500
        self.env = gym.make(env_name, initial_stacks=stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    def equity_self_improvement(self, improvement_rounds):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        calling = [.1, .2, .3, .4, .5, .6]
        betting = [.2, .3, .4, .5, .6, .7]

        for improvement_round in range(improvement_rounds):
            env_name = 'neuron_poker-v0'
            stack = 500
            self.env = gym.make(env_name, initial_stacks=stack, render=self.render)
            for i in range(6):
                self.env.add_player(EquityPlayer(name=f'Equity/{calling[i]}/{betting[i]}',
                                                 min_call_equity=calling[i],
                                                 min_bet_equity=betting[i]))

            for _ in range(self.num_episodes):
                self.env.reset()
                self.winner_in_episodes.append(self.env.winner_ix)

            league_table = pd.Series(self.winner_in_episodes).value_counts()
            best_player = int(league_table.index[0])
            print(league_table)
            print(f"Best Player: {best_player}")

            # self improve:
            self.log.info(f"Self improvment round {improvement_round}")
            for i in range(6):
                calling[i] = np.mean([calling[i], calling[best_player]])
                self.log.info(f"New calling for player {i} is {calling[i]}")
                betting[i] = np.mean([betting[i], betting[best_player]])
                self.log.info(f"New betting for player {i} is {betting[i]}")

    def dqn_train_keras_rl(self):
        """Implementation of kreras-rl deep q learing."""
        env_name = 'neuron_poker-v0'
        stack = 2000
        env = gym.make(env_name, initial_stacks=stack, funds_plot=self.funds_plot, render=self.render,
                       use_cpp_montecarlo=self.use_cpp_montecarlo)

        np.random.seed(123)
        env.seed(123)
        #        env.add_player(EquityPlayer(name='equity/50/70', min_call_equity=.5, min_bet_equity=.7))
        # env.add_player(RandomPlayer())
        # env.add_player(RandomPlayer())
        # env.add_player(RandomPlayer())
        # env.add_player(PlayerShell(name='keras-rl-1', stack_size=stack), range=0.9)  # shell is used for callback to keras rl
        # env.add_player(PlayerShell(name='keras-rl-2', stack_size=stack), range=0.9)  # shell is used for callback to keras rl
        # env.add_player(PlayerShell(name='keras-rl-3', stack_size=stack), range=0.9)  # shell is used for callback to keras rl
        # env.add_player(PlayerShell(name='keras-rl-4', stack_size=stack), range=0.9)  # shell is used for callback to keras rl
        # env.add_player(PlayerShell(name='keras-rl-5', stack_size=stack), range=0.9)  # shell is used for callback to keras rl
        # env.add_player(PlayerShell(name='keras-rl-6', stack_size=stack), range=0.9)  # shell is used for callback to keras rl
        # env.add_player(PlayerShell(name='keras-rl-7', stack_size=stack), range=0.9)  # shell is used for callback to keras rl
        env.add_player(PlayerShell(name='LJY', stack_size=stack, range=0.33))  # shell is used for callback to keras rl
        # dqn = DQNPlayer(name='DQN-1',stack_size=2000, range=0.9, env=env , load_model=None)
        # env.add_player(dqn)
        env.add_player(RandomPlayer(name='Random-1',range=1))
        # env.add_player(RandomPlayer(name='Random-2',range=1))
        # env.add_player(RandomPlayer(name='Random-3',range=1))
        # env.add_player(RandomPlayer(name='Random-4',range=1))
        # env.add_player(RandomPlayer(name='Random-5',range=1))
        # env.add_player(RandomPlayer(name='Random-6',range=1))
        # env.add_player(RandomPlayer(name='Random-7',range=1))
        # env.add_player(DQNPlayer(name='DQN-2',stack_size=2000, range=0.9, env=env , load_model=None))
        # env.add_player(DQNPlayer(name='DQN-3',stack_size=2000, range=0.9, env=env , load_model=None))
        # env.add_player(DQNPlayer(name='DQN-4',stack_size=2000, range=0.9, env=env , load_model=None))
        # env.add_player(DQNPlayer(name='DQN-5',stack_size=2000, range=0.9, env=env , load_model=None))
        # env.add_player(DQNPlayer(name='DQN-6',stack_size=2000, range=0.9, env=env , load_model=None))
        # env.add_player(DQNPlayer(name='DQN-7',stack_size=2000, range=0.9, env=env , load_model=None))
        # env.add_player(DQNPlayer(name='DQN-8',stack_size=2000, range=0.9, env=env , load_model=None))
        env.reset()
        # print(env.players[0].range)
        # print(env.players[1].range)
        # print(env.players[2].range)
        # print(env.players[3].range)
        # print(env.players[4].range)
        # print(env.players[5].range)
        dqn = DQNPlayer()
        # dqn.initiate_agent(env,load_model='3dqn_vs_3rd')
        dqn.initiate_agent(env)
        dqn.train(ckpt_name='LJY')

    def dqn_play_keras_rl(self):
        """Create 6 players, one of them a trained DQN"""
        env_name = 'neuron_poker-v0'
        stack = 2000
        self.env = gym.make(env_name, initial_stacks=stack, render=self.render)
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=.5))
        self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(PlayerShell(name='keras-rl', stack_size=stack))

        dqn = DQNPlayer(load_model='3dqn_vs_3rd', env=self.env)
        dqn.play(nb_episodes=self.num_episodes, render=self.render)

    def dqn_train_custom_q1(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        env_name = 'neuron_poker-v0'
        stack = 500
        self.env = gym.make(env_name, initial_stacks=stack, render=self.render)
        # self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        # self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        # self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        # self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())
        self.env.add_player(Custom_Q1(name='Deep_Q1'))

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")


if __name__ == '__main__':
    run = Runner(render=True,num_episodes=1, use_cpp_montecarlo=False,
                    funds_plot=False)
    run.dqn_train_keras_rl()
    # run.key_press_agents()
    # command_line_parser()