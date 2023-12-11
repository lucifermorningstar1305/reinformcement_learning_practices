"""
An implementation of the Monte-Carlo Control method without Explorling Starts
"""

from typing import Dict, List, Tuple
import numpy as np

from gridworld import GridWorld, standard_grid, negative_grid
from utils import print_policy, print_values

ACTION_SPACE = ("U", "D", "L", "R")
NUM_TRIALS = 10_000
GAMMA = 0.9


def epsilon_greedy(policy: Dict, s: Tuple, eps: float = 0.1):
    """Function to greedily select an action following epsilon-greedy algorithm"""
    p = np.random.random()

    if p < (1 - eps):
        return policy[s]
    else:
        return np.random.choice(ACTION_SPACE)


def play_game(
    policy: Dict, grid: GridWorld, max_steps: int = 20
) -> Tuple[list, list, list]:
    """Function to play one episode of the game"""

    s = grid.reset()
    a = epsilon_greedy(policy=policy, s=s)

    states = [s]
    actions = [a]
    rewards = [0]

    for _ in range(max_steps):
        r = grid.move(a)
        s = grid.current_state()

        states.append(s)
        rewards.append(r)

        if grid.game_over():
            break
        else:
            a = epsilon_greedy(policy=policy, s=s)
            actions.append(a)

    return states, actions, rewards


def online_mean_calc(prev_mean: float, cur_val: float, n_samples: int) -> float:
    """Function to calculate the mean in online fashion"""

    lr = 1 / n_samples
    new_mean = prev_mean + lr * (cur_val - prev_mean)

    return new_mean


def max_res(action_values: Dict) -> Tuple:
    max_val = max(action_values.values())
    max_action = [k for k, v in action_values.items() if v == max_val]

    return np.random.choice(max_action), max_val


if __name__ == "__main__":
    grid = standard_grid()
    # grid = negative_grid(step_cost=-0.1)

    # Initialize random policy
    policy = dict()
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ACTION_SPACE)

    print("Initial Policy: ")
    print_policy(policy=policy, grid=grid)
    print("\n\n")

    # Initialize Q-table
    Q = dict()
    samples_count = dict()

    for s in grid.get_all_states():
        if not grid.is_terminal(s):
            Q[s] = {}
            samples_count[s] = {}
            for a in ACTION_SPACE:
                Q[s][a] = 0
                samples_count[s][a] = 0
        else:
            pass

    for _ in range(NUM_TRIALS):
        states, actions, rewards = play_game(policy=policy, grid=grid)
        G = 0
        T = len(states)

        state_actions = list(zip(states, actions))

        for t in range(T - 2, -1, -1):
            G = rewards[t + 1] + GAMMA * G
            s = states[t]
            a = actions[t]

            if (s, a) not in state_actions[:t]:
                samples_count[s][a] += 1
                Q[s][a] = online_mean_calc(
                    prev_mean=Q[s][a], cur_val=G, n_samples=samples_count[s][a]
                )
                policy[s] = max_res(Q[s])[0]

    ###########################################
    ########## CALC VALUE FUNCTION ############
    ###########################################

    V = dict()
    for s in Q.keys():
        if not grid.is_terminal(s):
            V[s] = max_res(Q[s])[1]
        else:
            V[s] = 0

    print("Final Values of the state: ")
    print_values(values=V, grid=grid)
    print("\n\n")

    print("Final Policy:")
    print_policy(policy=policy, grid=grid)
    print("\n\n")
