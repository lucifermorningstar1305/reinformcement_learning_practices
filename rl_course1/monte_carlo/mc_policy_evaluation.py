from typing import Dict

import numpy as np

from gridworld import GridWorld, standard_grid
from utils import print_policy, print_values

EPS = 1e-3
ACTION_SPACE = ("U", "D", "L", "R")
GAMMA = 0.9


def play_game(grid: GridWorld, policy: Dict, max_steps: int = 20):
    start_states = list(grid.get_all_states())
    start_idx = np.random.choice(range(len(start_states)))

    grid.set_state(start_states[start_idx])
    s = grid.current_state()

    states = [s]
    rewards = [0]

    steps = 0

    while not grid.game_over():
        a = policy[s]
        r = grid.move(a)

        s = grid.current_state()

        rewards.append(r)
        states.append(s)

        steps += 1
        if steps >= max_steps:
            break

    return states, rewards


if __name__ == "__main__":
    grid = standard_grid()

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

    print_policy(policy=policy, grid=grid)
    print("\n\n")

    V = dict()
    returns = dict()

    for s in grid.get_all_states():
        if s in grid.actions:  # Checks if the current state is a terminal state
            returns[s] = list()
        else:
            V[s] = 0  # V[s] sh

    for _ in range(100):
        states, rewards = play_game(grid=grid, policy=policy, max_steps=100)
        G = 0
        T = len(states)
        for t in range(T - 2, -1, -1):
            G = rewards[t + 1] + GAMMA * G
            s = states[t]

            if s not in states[:t]:  # Using first-visit Monte Carlo
                returns[s].append(G)
                V[s] = np.mean(returns[s])

    print("Values:")
    print_values(values=V, grid=grid)
    print("\n\n")
    print("Policy")
    print_policy(policy=policy, grid=grid)
