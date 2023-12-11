"""
An implementation of the temporal difference prediction algorithm.

"""
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from gridworld import standard_grid, negative_grid
from utils import print_policy, print_values


GAMMA = 0.9
ALPHA = 0.1
ACTION_SPACE = ("U", "D", "L", "R")
NUM_EPISODES = 10_000


def epsilon_greedy(policy: Dict, state: Tuple, eps: float = 0.1):
    p = np.random.random()

    if p < (1 - eps):
        return policy[state]
    else:
        return np.random.choice(ACTION_SPACE)


if __name__ == "__main__":
    grid = standard_grid()

    # Initialise policy:
    policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "R",
        (2, 1): "R",
        (2, 2): "R",
        (2, 3): "U",
    }

    # Initialise State-Values
    V = dict()
    state_visit_count = dict()

    for s in grid.get_all_states():
        V[s] = 0
        state_visit_count[s] = 0

    deltas = list()

    for _ in range(NUM_EPISODES):
        s = grid.reset()
        delta = 0

        while not grid.game_over():
            a = epsilon_greedy(policy=policy, state=s)
            r = grid.move(a)
            s_next = grid.current_state()

            old_v = V[s]
            V[s] = V[s] + ALPHA * (r + GAMMA * V[s_next] - V[s])
            delta = max(delta, np.abs(old_v - V[s]))
            state_visit_count[s] += 1
            s = s_next

        deltas.append(delta)

    plt.plot(deltas)
    plt.show()

    print("Values:")
    print_values(values=V, grid=grid)
    print("\n\n")

    print("Policy:")
    print_policy(policy=policy, grid=grid)
    print("\n\n")

    print("State Visit Count:")
    print_values(values=state_visit_count, grid=grid)
