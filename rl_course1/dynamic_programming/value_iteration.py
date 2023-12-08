from typing import Dict, Optional, Tuple

import numpy as np

from gridworld import WindyGridWorld, windy_grid, windy_grid_penalized
from iterative_policy import print_policy, print_values


THRESHOLD = 1e-3
GAMMA = 0.9
ACTION_SPACE = ("U", "D", "L", "R")


def get_state_trans_probs(grid: WindyGridWorld) -> Tuple:
    """Function to obtain the rewards and the state-transition probabilities."""

    state_transition = dict()
    rewards = dict()

    for (s, a), v in grid.state_transition.items():
        for s_next, prob in v.items():
            state_transition[(s, a, s_next)] = prob
            rewards[(s, a, s_next)] = grid.rewards.get(s_next, 0.0)

    return state_transition, rewards


if __name__ == "__main__":
    grid = windy_grid()
    state_transition, rewards = get_state_trans_probs(grid)

    # initialize state-value for all states
    V = dict()
    for s in grid.get_all_states():
        V[s] = 0.0

    #############################################################################
    ########################### VALUE ITERATION #################################
    #############################################################################

    # ********************************************** #
    # *************** POLICY EVALUATION************* #
    # ********************************************** #

    idx = 0
    while True:
        delta = 0

        for s in grid.get_all_states():
            if not grid.game_over(s):
                v_old = V[s]
                v_new = float("-inf")

                for a in ACTION_SPACE:
                    v = 0
                    for s_next in grid.get_all_states():
                        r = rewards.get((s, a, s_next), 0.0)
                        v += state_transition.get((s, a, s_next), 0.0) * (
                            r + GAMMA * V[s_next]
                        )

                    if v > v_new:
                        v_new = v

                V[s] = v_new
                delta = max(delta, np.abs(v_old - V[s]))

        if delta < THRESHOLD:
            break

        idx += 1

    policy = dict()

    for s in grid.actions.keys():
        best_a = None
        best_value = float("-inf")

        for a in ACTION_SPACE:
            v = 0
            for s_next in grid.get_all_states():
                r = rewards.get((s, a, s_next), 0.0)
                v += state_transition.get((s, a, s_next), 0.0) * (r + GAMMA * V[s_next])

            if v > best_value:
                best_value = v
                best_a = a

        policy[s] = best_a

    print("Values:")
    print_values(V=V, g=grid)
    print("\n\n")

    print("Policy:")
    print_policy(P=policy, g=grid)
