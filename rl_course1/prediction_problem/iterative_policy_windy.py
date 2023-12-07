from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from gridworld import WindyGridWorld, windy_grid

THRESHOLD = 1e-3
ACTION_SPACE = ("U", "L", "D", "R")
GAMMA = 0.9


def print_values(V: Dict, g: WindyGridWorld):
    """Function to print the values of each state for a given gridworld object."""

    for row in range(g.nrows):
        print("-" * 10)
        for col in range(g.ncols):
            v = V.get((row, col), 0)
            if v >= 0:
                print(f" {v:.2f}|", end="")
            else:
                print(f"{v:.2f}|", end="")

        print("")


def print_policy(P: Dict, g: WindyGridWorld):
    """Function to print the policy for the WindyGridWorld"""

    for row in range(g.nrows):
        print("-" * 10)
        for col in range(g.ncols):
            act = P.get((row, col), "None")

            print(f" {act} |", end="")

        print("")


if __name__ == "__main__":
    rewards = dict()

    grid = windy_grid()

    ############################################
    ########## STATE-TRANS & REWARDS ###########
    ############################################
    state_trans = dict()
    rewards = dict()

    for (s, a), v in grid.state_transition.items():
        for s_next, prob in v.items():
            state_trans[(s, a, s_next)] = prob
            rewards[(s, a, s_next)] = grid.rewards.get(s_next, 0.0)

    ###################################
    ############ FIXED POLICY #########
    ###################################

    policy = {
        (2, 0): {"U": 0.5, "R": 0.5},
        (1, 0): {"U": 1.0},
        (0, 0): {"R": 1.0},
        (0, 1): {"R": 1.0},
        (0, 2): {"R": 1.0},
        (1, 2): {"U": 1.0},
        (2, 1): {"R": 1.0},
        (2, 2): {"U": 1.0},
        (2, 3): {"L": 1.0},
    }

    print_policy(P=policy, g=grid)
    print("\n\n")

    state_value = dict()

    #####################################
    ######## INITIALISE V(S)=0 #########
    ####################################

    for s in grid.get_all_states():
        state_value[s] = 0

    all_states = grid.get_all_states()

    idx = 0
    while True:
        delta = 0
        for s in all_states:
            if not grid.game_over(state=s):
                v_old = state_value[s]
                v_new = 0
                for act in ACTION_SPACE:
                    for s_next in all_states:
                        action_prob = policy[s].get(act, 0.0)
                        r = rewards.get((s, a, s_next), 0.0)

                        ######################################################################
                        ########### v += pi(a|s) * p(s'|s, a) * [r + gamma * V[s']] ##########
                        ######################################################################
                        v_new += (
                            action_prob
                            * state_trans.get((s, act, s_next), 0.0)
                            * (r + GAMMA * state_value[s_next])
                        )
                state_value[s] = v_new
                delta = max(delta, np.abs(v_old - state_value[s]))

        print(f"Iteration : {idx}, delta: {delta}")
        print_values(V=state_value, g=grid)
        idx += 1

        if delta < THRESHOLD:
            break

        print("\n\n")
