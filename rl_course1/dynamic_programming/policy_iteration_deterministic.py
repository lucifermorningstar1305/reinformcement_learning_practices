from typing import Dict, Optional

import numpy as np
from gridworld import GridWorld, standard_grid
from iterative_policy import print_policy, print_values


THRESHOLD = 1e-3
GAMMA = 0.9
ACTION_SPACE = ("U", "D", "L", "R")


def get_state_transition(grid: GridWorld):
    """Function to initialize the reward and state_transition"""

    state_transition = dict()
    rewards = dict()

    for row in range(grid.nrows):
        for col in range(grid.ncols):
            s = (row, col)

            if not grid.game_over(s):
                for a in ACTION_SPACE:
                    s_next = grid.get_next_state(state=s, action=a)

                    state_transition[(s, a, s_next)] = 1
                    if s_next in grid.rewards:
                        rewards[(s, a, s_next)] = grid.rewards[s_next]

    return state_transition, rewards


def evaluate_policy(grid: GridWorld, policy: Dict, initV: Optional[Dict] = None):
    """Function to perform Policy Evaluation"""

    V = dict()
    if initV is None:
        for s in grid.get_all_states():
            V[s] = 0

    else:
        V = initV

    idx = 0

    while True:
        delta = 0
        for s in grid.get_all_states():
            if not grid.game_over(s):
                v_old = V[s]
                v_new = 0

                for a in ACTION_SPACE:
                    for s_next in grid.get_all_states():
                        action_prob = 1.0 if policy.get(s) == a else 0.0
                        r = rewards.get((s, a, s_next), 0.0)

                        v_new += (
                            action_prob
                            * state_transition.get((s, a, s_next), 0.0)
                            * (r + GAMMA * V[s_next])
                        )

                V[s] = v_new
                delta = max(delta, np.abs(v_old - V[s]))

        idx += 1
        if delta < THRESHOLD:
            break

    return V


if __name__ == "__main__":
    grid = standard_grid()
    state_transition, rewards = get_state_transition(grid=grid)

    V = None

    # Randomly Generate a policy
    policy = dict()
    for s in grid.get_all_states():
        policy[s] = np.random.choice(ACTION_SPACE)

    print("INITIAL POLICY:")
    print_policy(P=policy, g=grid)
    print("\n\n")

    #######################################################################
    ##################### POLICY ITERATION/IMPROVEMENT ####################
    #######################################################################
    idx = 0
    while True:
        V = evaluate_policy(grid=grid, policy=policy, initV=V)

        is_converged = True

        for s in grid.get_all_states():
            old_a = policy[s]
            new_a = None
            best_val = float("-inf")

            for a in ACTION_SPACE:
                v = 0
                for s_next in grid.get_all_states():
                    state_transition_prob = state_transition.get((s, a, s_next), 0.0)
                    r = rewards.get((s, a, s_next), 0.0)

                    v += state_transition_prob * (r + GAMMA * V[s_next])

                if v > best_val:
                    best_val = v
                    new_a = a

            policy[s] = new_a
            if new_a != old_a:
                is_converged = False

        print(f"Iteration : {idx}")
        print_values(V=V, g=grid)
        print("\n\n")

        if is_converged:
            break

        idx += 1

    print("NEW POLICY:")
    print_policy(P=policy, g=grid)
    print("\n\n")

    print("STATE-VALUES")
    print_values(V=V, g=grid)
