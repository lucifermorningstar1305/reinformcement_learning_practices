import numpy as np
from gridworld import standard_grid, negative_grid
from utils import print_policy, print_values


ACTION_SPACE = ("U", "D", "L", "R")
THRESHOLD = 1e-5
GAMMA = 0.9

if __name__ == "__main__":
    grid = negative_grid(step_cost=-0.1)
    print("Rewards:")
    print_values(values=grid.rewards, grid=grid)
    print("\n\n")

    policy = {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "U",
        (2, 1): "L",
        (2, 2): "U",
        (2, 3): "U",
    }

    V = dict()

    for s in grid.get_all_states():
        V[s] = 0

    state_transition = dict()
    rewards = dict()
    for row in range(grid.nrows):
        for col in range(grid.ncols):
            s = (row, col)

            if not grid.is_terminal(s):
                for a in ACTION_SPACE:
                    s_next = grid.get_next_state(state=s, action=a)
                    state_transition[(s, a, s_next)] = 1
                    if s_next in grid.rewards:
                        rewards[(s, a, s_next)] = grid.rewards[s_next]

    while True:
        delta = 0
        for s in grid.get_all_states():
            if not grid.is_terminal(s):
                V_old = V[s]
                V_new = 0
                for a in ACTION_SPACE:
                    for s_next in grid.get_all_states():
                        action_prob = 1.0 if policy.get(s) == a else 0.0
                        r = rewards.get((s, a, s_next), 0.0)
                        p = state_transition.get((s, a, s_next), 0)

                        V_new += action_prob * p * (r + GAMMA * V[s_next])

                V[s] = V_new
                delta = max(delta, np.abs(V_old - V[s]))

        if delta < THRESHOLD:
            break

    print("Values:")
    print_values(values=V, grid=grid)
