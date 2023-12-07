from typing import Dict, Tuple

import numpy as np


class GridWorld(object):
    def __init__(self, nrows: int, ncols: int, start_pos: Tuple):
        self.nrows = nrows
        self.ncols = ncols
        self.pos_i, self.pos_j = start_pos

    def set_rewards(self, rewards: Dict, actions: Dict):
        """Function to set rewards and action space.

        Parameters:
        ------------
        rewards: {(i, j): reward_value}
        actions: {(i, j): list of actions}
        """
        self.rewards = rewards
        self.actions = actions

    def set_state(self, state: Tuple):
        """Function to set the state for the agent"""
        self.pos_i, self.pos_j = state

    def get_next_state(self, state: Tuple, action: str):
        """Function to change a state based on current state and action"""
        pos_i, pos_j = state

        if action in self.actions.get((pos_i, pos_j), ()):
            if action == "D":
                pos_i += 1
            elif action == "R":
                pos_j += 1
            elif action == "U":
                pos_i -= 1
            elif action == "L":
                pos_j -= 1

        return (pos_i, pos_j)

    def get_all_states(self):
        """Function to return the complete space"""
        return set(self.actions.keys()).union(set(self.rewards.keys()))

    def move(self, action: str):
        """Function to move the agent based on an action"""
        if action in self.actions.get((self.pos_i, self.pos_j), ()):
            if action == "D":
                self.pos_i += 1
            elif action == "R":
                self.pos_j += 1
            elif action == "U":
                self.pos_i -= 1
            elif action == "L":
                self.pos_j -= 1

        new_state = (self.pos_i, self.pos_j)
        return new_state, self.actions[new_state], self.rewards.get(new_state, 0)

    def undo_move(self, action: str):
        """Function to undo a move of an agent"""
        if action == "D":
            self.pos_i -= 1
        elif action == "R":
            self.pos_j -= 1
        elif action == "U":
            self.pos_i += 1
        elif action == "L":
            self.pos_j += 1

        assert (self.pos_i, self.pos_j) in self.get_all_states()

    def current_state(self):
        """Function to obtain the current state of the agent"""
        return (self.pos_i, self.pos_j)

    def game_over(self, state: Tuple):
        """Function to check if the game is over or not"""
        return state not in self.actions


class WindyGridWorld(object):
    def __init__(self, nrows: int, ncols: int, start_pos: Tuple):
        self.nrows = nrows
        self.ncols = ncols
        self.pos_i, self.pos_j = start_pos

    def set_rewards(self, rewards: Dict, actions: Dict, state_trans: Dict):
        """Function to set rewards and action space.

        Parameters:
        ------------
        rewards: {(i, j): reward_value}
        actions: {(i, j): list of actions}
        """
        self.rewards = rewards
        self.actions = actions
        self.state_transition = state_trans

    def set_state(self, state: Tuple):
        """Function to set the state for the agent"""
        self.pos_i, self.pos_j = state

    def get_all_states(self):
        """Function to return the complete space"""
        return set(self.actions.keys()).union(set(self.rewards.keys()))

    def move(self, action: str):
        """Function to move the agent based on an action"""

        s = (self.pos_i, self.pos_j)
        s_next_probs = self.state_transition[(s, action)]
        possible_s_nexts = list(s_next_probs.keys())
        probs = list(s_next_probs.values())

        s_next = np.random.choice(possible_s_nexts, p=probs)

        # Update current state
        self.pos_i, self.pos_j = s_next

        reward = self.rewards.get(s_next, 0)
        return s_next, reward

    def current_state(self):
        """Function to obtain the current state of the agent"""
        return (self.pos_i, self.pos_j)

    def game_over(self, state: Tuple):
        """Function to check if the game is over or not"""
        return state not in self.actions


def standard_grid():
    """Function to setup a standard gridworld."""
    grid = GridWorld(3, 4, (2, 0))

    actions = {
        (0, 0): ("D", "R"),
        (0, 1): ("L", "R"),
        (0, 2): ("L", "D", "R"),
        (1, 0): ("U", "D"),
        (1, 2): ("U", "D", "R"),
        (2, 0): ("U", "R"),
        (2, 1): ("L", "R"),
        (2, 2): ("U", "L", "R"),
        (2, 3): ("U", "L"),
    }

    rewards = {(0, 3): 1, (1, 3): -1}

    grid.set_rewards(rewards=rewards, actions=actions)
    return grid


def windy_grid():
    """Function to setup a standard windy grid."""

    wgrid = WindyGridWorld(3, 4, (2, 0))
    rewards = {(0, 3): 1, (1, 3): -1}
    actions = {
        (0, 0): ("D", "R"),
        (0, 1): ("L", "R"),
        (0, 2): ("L", "D", "R"),
        (1, 0): ("U", "D"),
        (1, 2): ("U", "D", "R"),
        (2, 0): ("U", "R"),
        (2, 1): ("L", "R"),
        (2, 2): ("U", "L", "R"),
        (2, 3): ("U", "L"),
    }

    # State transition : p(s' | s, a)
    state_trans = {
        ((2, 0), "U"): {(1, 0): 1.0},
        ((2, 0), "D"): {(2, 0): 1.0},
        ((2, 0), "L"): {(2, 0): 1.0},
        ((2, 0), "R"): {(2, 1): 1.0},
        ((1, 0), "U"): {(0, 0): 1.0},
        ((1, 0), "D"): {(2, 0): 1.0},
        ((1, 0), "L"): {(1, 0): 1.0},
        ((1, 0), "R"): {(1, 0): 1.0},
        ((0, 0), "U"): {(0, 0): 1.0},
        ((0, 0), "D"): {(1, 0): 1.0},
        ((0, 0), "L"): {(0, 0): 1.0},
        ((0, 0), "R"): {(0, 1): 1.0},
        ((0, 1), "U"): {(0, 1): 1.0},
        ((0, 1), "D"): {(0, 1): 1.0},
        ((0, 1), "L"): {(0, 0): 0.85, (0, 2): 0.15},
        ((0, 1), "R"): {(0, 2): 0.95, (0, 0): 0.05},
        ((0, 2), "U"): {(0, 2): 1.0},
        ((0, 2), "D"): {(1, 2): 1.0},
        ((0, 2), "L"): {(0, 1): 0.7, (0, 3): 0.3},
        ((0, 2), "R"): {(0, 3): 0.95, (0, 1): 0.05},
        ((2, 1), "U"): {(2, 1): 1.0},
        ((2, 1), "D"): {(2, 1): 1.0},
        ((2, 1), "L"): {(2, 0): 1.0},
        ((2, 1), "R"): {(2, 2): 1.0},
        ((2, 2), "U"): {(1, 2): 1.0},
        ((2, 2), "D"): {(2, 2): 1.0},
        ((2, 2), "L"): {(2, 1): 1.0},
        ((2, 2), "R"): {(2, 3): 1.0},
        ((1, 2), "U"): {(0, 2): 1.0},
        ((1, 2), "D"): {(2, 2): 1.0},
        ((1, 2), "L"): {(1, 2): 1.0},
        ((1, 2), "R"): {(1, 3): 0.5, (2, 3): 0.5},
    }

    wgrid.set_rewards(rewards=rewards, actions=actions, state_trans=state_trans)

    return wgrid
