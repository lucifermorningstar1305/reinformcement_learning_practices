"""
An implementation of the function approximation using TD-Learning 
"""

from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler

from gridworld import standard_grid, negative_grid, GridWorld
from utils import print_values, print_policy


GAMMA = 0.9
ALPHA = 0.01
ACTION_SPACE = ("U", "D", "L", "R")
NUM_EPISODES = 20_000


def epsilon_greedy(policy: Dict, state: Tuple, eps: float = 0.1) -> str:
    """Function to select an action greedily following the epsilon-greedy approach."""

    p = np.random.random()

    if p < (1 - eps):
        return policy[state]

    else:
        return np.random.choice(ACTION_SPACE)


def gather_samples(grid: GridWorld, n_episodes: int = 10_000) -> List:
    """Function to randomly gather sample states from the grid environment."""

    states = list()
    for _ in range(n_episodes):
        s = grid.reset()
        states.append(s)

        while not grid.game_over():
            a = np.random.choice(ACTION_SPACE)
            r = grid.move(a)
            s_next = grid.current_state()

            states.append(s_next)
            s = s_next

    return states


class Model:
    def __init__(self, grid: GridWorld):
        samples = gather_samples(grid=grid)

        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)

        dims = self.featurizer.n_components

        self.w = np.zeros(dims)

    def predict(self, state: Tuple) -> np.ndarray:
        x = self.featurizer.transform([state])[0]
        return x @ self.w

    def grad(self, state: Tuple) -> np.ndarray:
        return self.featurizer.transform([state])[0]


if __name__ == "__main__":
    grid = standard_grid()

    print("Rewards:")
    print_values(values=grid.rewards, grid=grid)
    print("\n\n")

    greedy_policy = {
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

    model = Model(grid=grid)
    mse_per_episode = list()

    for episode in range(NUM_EPISODES):
        if episode % 1000:
            print(f"Episode {episode} in Progress....")
        s = grid.reset()
        Vs = model.predict(s)  # w^T.x

        episode_err = 0
        n_steps = 0

        while not grid.game_over():
            a = epsilon_greedy(policy=greedy_policy, state=s)
            r = grid.move(a)
            s_next = grid.current_state()

            if grid.is_terminal(s_next):
                y = r
            else:
                Vs_next = model.predict(s_next)
                y = r + GAMMA * Vs_next

            grad = model.grad(s)
            error = y - Vs
            model.w += ALPHA * error * grad

            n_steps += 1
            episode_err += error**2

            s = s_next
            Vs = Vs_next

        mse = episode_err / n_steps
        mse_per_episode.append(mse)

    plt.plot(mse_per_episode)
    plt.title("MSE Per Episode")
    plt.show()

    V = dict()

    for s in grid.get_all_states():
        if not grid.is_terminal(s):
            V[s] = model.predict(s)

        else:
            V[s] = 0

    print("Values:")
    print_values(values=V, grid=grid)
    print("\n\n")

    print("Greedy Policy:")
    print_policy(policy=greedy_policy, grid=grid)
    print("\n\n")
