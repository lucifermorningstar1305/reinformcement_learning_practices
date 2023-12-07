from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld, standard_grid

THRESHOLD = 1e-3
ACTION_SPACE = ("U", "L", "D", "R")
GAMMA = 0.9


def print_values(V: Dict, g: GridWorld):
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


def print_policy(P: Dict, g: GridWorld):
    """Function to print the policy for the gridworld"""

    for row in range(g.nrows):
        print("-" * 10)
        for col in range(g.ncols):
            act = P[(row, col)] if not g.game_over((row, col)) else "None"

            print(f" {act} |", end="")

        print("")


if __name__ == "__main__":
    state_transition = dict()
    rewards = dict()

    grid = standard_grid()

    ###############################################
    ###### GENERATE THE STATE TRANSITION ##########
    ################ p(s'|s, a) ###################
    ################################################
    for row in range(grid.nrows):
        for col in range(grid.ncols):
            s = (row, col)

            if not grid.game_over(s):
                for act in ACTION_SPACE:
                    s_next = grid.get_next_state(state=s, action=act)
                    state_transition[(s, act, s_next)] = 1
                    if s_next in grid.rewards:
                        rewards[(s, act, s_next)] = grid.rewards[s_next]

    ###################################
    ############ FIXED POLICY #########
    ###################################

    fixed_policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "U",
        (2, 1): "R",
        (2, 2): "U",
        (2, 3): "L",
    }

    print_policy(P=fixed_policy, g=grid)
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
                        action_prob = 1 if fixed_policy.get(s) == act else 0
                        r = rewards.get((s, act, s_next), 0)

                        ######################################################################
                        ########### v += pi(a|s) * p(s'|s, a) * [r + gamma * V[s']] ##########
                        ######################################################################
                        v_new += (
                            action_prob
                            * state_transition.get((s, act, s_next), 0)
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
