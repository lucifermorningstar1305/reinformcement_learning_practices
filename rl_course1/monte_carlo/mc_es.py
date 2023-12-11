"""
Implementation of the Monte-Carlo Exploration starts method.
"""

from typing import Dict, Tuple
import numpy as np

from gridworld import standard_grid, GridWorld
from utils import print_policy, print_values


NUM_TRIALS = 10_000
GAMMA = 0.9
ACTION_SPACE = ("U", "D", "L", "R")


def play_episode(
    grid: GridWorld, policy: Dict, max_steps: int = 20
) -> Tuple[list, list, list]:
    """Function to simulate a playing of an episode."""

    start_states = list(grid.actions.keys())
    states_start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[states_start_idx])

    s = grid.current_state()
    a = np.random.choice(ACTION_SPACE)

    states = [s]
    actions = [a]
    rewards = [0]

    for _ in range(max_steps):
        r = grid.move(a)
        s = grid.current_state()

        rewards.append(r)
        states.append(s)

        if grid.game_over():
            break
        else:
            a = policy[s]
            actions.append(a)

    return states, actions, rewards


def online_mean_calc(prev_mean: float, x: float, n_samples: int) -> float:
    """Function to calculate the mean in an online way"""
    return prev_mean + (x - prev_mean) / n_samples


def max_res(action_values: Dict):
    """Function which simulates the argmax Q(s, a)"""

    max_vals = max(action_values.values())
    max_keys = [k for k, v in action_values.items() if v == max_vals]

    return np.random.choice(max_keys), max_vals


if __name__ == "__main__":
    grid = standard_grid()

    policy = dict()

    # Random policy generation
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)

    print("initial policy:")
    print_policy(policy=policy, grid=grid)
    print("\n\n")

    # Q-Table initialization
    Q = dict()
    sample_counts = dict()

    for s in grid.get_all_states():
        Q[s] = {}
        sample_counts[s] = {}
        for a in ACTION_SPACE:
            if grid.is_terminal(s):
                pass

            else:
                Q[s][a] = 0
                sample_counts[s][a] = 0

    for _ in range(NUM_TRIALS):
        states, actions, rewards = play_episode(grid=grid, policy=policy, max_steps=100)
        state_actions = list(zip(states, actions))

        G = 0
        T = len(states)

        for t in range(T - 2, -1, -1):
            G = rewards[t + 1] + GAMMA * G

            s = states[t]
            a = actions[t]

            if (s, a) not in state_actions[:t]:
                sample_counts[s][a] += 1
                Q[s][a] = online_mean_calc(
                    prev_mean=Q[s][a], x=G, n_samples=sample_counts[s][a]
                )
                policy[s] = max_res(Q[s])[0]

    V = dict()
    for s in Q.keys():
        if not grid.is_terminal(s):
            V[s] = max_res(Q[s])[1]
        else:
            V[s] = 0

    print("final values: ")
    print_values(values=V, grid=grid)
    print("\n\n")

    print("final policy: ")
    print_policy(policy=policy, grid=grid)
