"""
An implementation of the Q-Learning algorithm.
"""

from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


from gridworld import standard_grid, negative_grid
from utils import print_policy, print_values


NUM_EPISODES = 10_000
GAMMA = 0.9
ALPHA = 0.1
ACTION_SPACE = ("U", "D", "L", "R")


def epsilon_greedy(Q: Dict, state: Tuple, eps: float = 0.1) -> str:
    """Function to implement the Epsilon-Greedy algorithm."""

    p = np.random.random()

    if p < eps:
        return np.random.choice(ACTION_SPACE)

    else:
        return argmax_action(Q[s])[0]


def argmax_action(action_values: Dict) -> Tuple[str, float]:
    """Function to calculate the optimal action and Q-value for a given state and Q-Function"""

    max_value = max(action_values.values())
    max_actions = [k for k, v in action_values.items() if v == max_value]

    return np.random.choice(max_actions), max_value


if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.1)

    print("Rewards:")
    print_values(values=grid.rewards, grid=grid)
    print("\n\n")

    # Initialize Q-function
    Q = dict()
    for s in grid.get_all_states():
        Q[s] = dict()
        for a in ACTION_SPACE:
            Q[s][a] = 0

    state_visit_count = dict()

    for s in grid.get_all_states():
        state_visit_count[s] = 0

    rewards_per_episode = list()

    for _ in range(NUM_EPISODES):
        s = grid.reset()
        reward = 0

        while not grid.game_over():
            a = epsilon_greedy(Q=Q, state=s)
            r = grid.move(a)
            s_next = grid.current_state()
            maxQ = argmax_action(Q[s_next])[1]

            reward += r
            state_visit_count[s] += 1
            Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * maxQ - Q[s][a])
            s = s_next

        rewards_per_episode.append(reward)

    plt.plot(rewards_per_episode)
    plt.show()

    policy = dict()
    V = dict()

    for s in Q.keys():
        opt_action, opt_val = argmax_action(Q[s])
        policy[s] = opt_action
        V[s] = opt_val

    print("Value Function")
    print_values(values=V, grid=grid)
    print("\n\n")

    print("Optimal Policy")
    print_policy(policy=policy, grid=grid)
    print("\n\n")

    print("State Visit Frequency")
    print_values(values=state_visit_count, grid=grid)
    print("\n\n")
