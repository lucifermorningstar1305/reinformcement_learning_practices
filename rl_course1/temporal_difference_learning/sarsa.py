""" 
An implementation of the SARSA algorithm.
"""

from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from utils import print_policy, print_values
from gridworld import GridWorld, standard_grid, negative_grid


GAMMA = 0.9
ALPHA = 0.1
NUM_EPISODES = 10_000
ACTION_SPACE = ("U", "D", "L", "R")


def epsilon_greedy(Q: Dict, state: Tuple, eps: float = 0.1):
    """Function to select an action greedily"""

    p = np.random.random()

    if p < eps:
        return np.random.choice(ACTION_SPACE)

    else:
        return argmax_action(Q[state])[0]


def argmax_action(action_values: Dict) -> Tuple[str, float]:
    """Function to select the optimal action."""

    max_value = max(action_values.values())
    max_action = [k for k, v in action_values.items() if v == max_value]

    return np.random.choice(max_action), max_value


if __name__ == "__main__":
    # grid = standard_grid()
    grid = negative_grid()

    # initialize Q values
    Q = dict()
    state_visit_count = dict()

    for s in grid.get_all_states():
        Q[s] = {}
        state_visit_count[s] = 0

        for a in ACTION_SPACE:
            Q[s][a] = 0

    # Initialize State-Values (V)
    V = dict()

    for s in grid.get_all_states():
        V[s] = 0

    deltas = list()
    reward_per_episode = list()

    for _ in range(NUM_EPISODES):
        s = grid.reset()
        a = epsilon_greedy(Q=Q, state=s)

        delta = 0
        episode_reward = 0
        while not grid.game_over():
            r = grid.move(a)
            s_next = grid.current_state()
            a_next = epsilon_greedy(Q=Q, state=s_next)

            Q_old = Q[s][a]
            state_visit_count[s] += 1
            episode_reward += r

            Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * Q[s_next][a_next] - Q[s][a])

            delta = max(delta, np.abs(Q_old - Q[s][a]))

            s = s_next
            a = a_next

            deltas.append(delta)

        reward_per_episode.append(episode_reward)

    plt.plot(deltas)
    plt.show()

    plt.plot(reward_per_episode)
    plt.title("Reward Per Episode")
    plt.show()

    policy = dict()
    V = dict()

    for s in Q.keys():
        optimal_a, optimal_val = argmax_action(Q[s])

        policy[s] = optimal_a
        V[s] = optimal_val

    print("State Values:")
    print_values(values=V, grid=grid)
    print("\n\n")

    print("Policy:")
    print_policy(policy=policy, grid=grid)
    print("\n\n")

    print("State Visit Count")
    print_values(values=state_visit_count, grid=grid)
    print("\n\n")
