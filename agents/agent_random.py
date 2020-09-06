"""Random player"""
import random

from gym_env.env import Action

autplay = True  # play automatically if played against keras-rl


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='Random',range=1):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True
        self.range = range

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL,Action.ALL_IN, Action.RAISE_200, Action.RAISE_250,Action.RAISE_300,
                                    Action.RAISE_350,Action.RAISE_400,Action.RAISE_450,Action.RAISE_500,Action.RAISE_550,
                                    Action.RAISE_600,Action.RAISE_10_POT,Action.RAISE_20_POT,Action.RAISE_30_POT,Action.RAISE_40_POT,
                                    Action.RAISE_50_POT,Action.RAISE_60_POT,Action.RAISE_70_POT,Action.RAISE_80_POT,Action.RAISE_90_POT,
                                    Action.RAISE_100_POT,Action.RAISE_125_POT,Action.RAISE_150_POT,Action.RAISE_175_POT,Action.RAISE_200_POT}

        possible_moves = this_player_action_space.intersection(set(action_space))
        action = random.choice(list(possible_moves))
        return action
