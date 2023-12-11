""" 
Implementation of the Function-Approximation Control Code using Q-Learning
"""

from typing import List, Tuple, Dict, Union, Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.kernel_approximation import RBFSampler
from gridworld import standard_grid, negative_grid, GridWorld
from utils import print_policy, print_values


GAMMA = 0.9
ACTION_SPACE = ["U", "D", "L", "R"]
ACTION2INT = {a: idx for idx, a in enumerate(ACTION_SPACE)}
ALPHA = 0.01

ONEHOT = np.eye(len(ACTION_SPACE))


def epsilon_greedy(model: Any, state: Tuple, eps: float = 0.1) -> str:
    """Function to select an action greedily based on epsilon-greedy algorithm."""

    p = np.random.random()

    if p < (1 - eps):
        vals = model.predict_all_actions(state)
        return ACTION_SPACE[np.argmax(vals)]
    else:
        return np.random.choice(ACTION_SPACE)


def get_one_hot(action: str) -> List:
    """Function to transform an action to one-hot encoded vector"""

    return ONEHOT[ACTION2INT[action], :]


def concat_state_action(state: Tuple, action: str) -> np.ndarray:
    """Function to concatenate the state-action pair"""

    one_hot = get_one_hot(action=action)
    return np.concatenate((state, one_hot))


def get_samples(grid: GridWorld, n_episodes: int = 1000) -> List:
    """Function to generate random set of state samples"""

    samples = list()

    for _ in range(n_episodes):
        s = grid.reset()

        while not grid.game_over():
            a = np.random.choice(ACTION_SPACE)

            state_action = concat_state_action(s, a)
            samples.append(state_action)

            r = grid.move(a)
            s = grid.current_state()

    return samples


class Model:
    def __init__(self, grid: GridWorld, n_epsiodes: int = 1000):
        samples = get_samples(grid=grid, n_episodes=n_epsiodes)

        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)

        dims = self.featurizer.n_components
        self.w = np.zeros(dims)

    def predict(self, state: Tuple, action: str) -> np.ndarray:
        sample = concat_state_action(state, action)
        x = self.featurizer.transform([sample])[0]
        return x @ self.w

    def grad(self, state: Tuple, action: str) -> np.ndarray:
        sample = concat_state_action(state, action)
        x = self.featurizer.transform([sample])[0]

        return x

    def predict_all_actions(self, state: Tuple) -> List:
        return [self.predict(state, a) for a in ACTION_SPACE]


if __name__ == "__main__":
    # grid = standard_grid()
    grid = negative_grid(step_cost=-0.1)

    print("Rewards:")
    print_values(values=grid.rewards, grid=grid)

    model = Model(grid=grid)

    n_episodes = 20_000
    mse_per_episode = list()
    rewards_per_episode = list()
    state_visit_count = dict()

    for episode in range(n_episodes):
        s = grid.reset()

        episode_err = 0
        n_steps = 0
        episode_reward = 0

        state_visit_count[s] = state_visit_count.get(s, 0) + 1

        while not grid.game_over():
            a = epsilon_greedy(model=model, state=s)
            r = grid.move(a)
            s_next = grid.current_state()

            state_visit_count[s_next] = state_visit_count.get(s_next, 0) + 1

            if grid.is_terminal(s_next):
                y = r

            else:
                values = model.predict_all_actions(state=s_next)
                y = r + GAMMA * np.max(values)

            grad = model.grad(state=s, action=a)
            err = y - model.predict(state=s, action=a)
            model.w += ALPHA * err * grad

            s = s_next

            episode_reward += r
            episode_err += err**2
            n_steps += 1

        episode_err /= n_steps

        mse_per_episode.append(episode_err)
        rewards_per_episode.append(episode_reward)

        if episode % 100 == 0:
            print(
                f"Episode-{episode} | MSE:{episode_err:.5f} | Reward: {episode_reward:.2f}"
            )

    plt.plot(mse_per_episode)
    plt.title("MSE per Episode")
    plt.show()

    plt.plot(rewards_per_episode)
    plt.title("Rewards per Episode")
    plt.show()

    V = dict()
    policy = dict()

    for s in grid.get_all_states():
        if not grid.is_terminal(s):
            values = model.predict_all_actions(s)
            V[s] = np.max(values)
            policy[s] = ACTION_SPACE[np.argmax(values)]

        else:
            V[s] = 0

    print("Values:")
    print_values(values=V, grid=grid)
    print("\n\n")

    print("Policy:")
    print_policy(policy=policy, grid=grid)
    print("\n\n")
