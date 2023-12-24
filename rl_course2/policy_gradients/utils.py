from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler


def plot_running_avg(rewards: np.ndarray, window: int = 100):
    """Function to plot the running average."""

    N = rewards.shape[0]
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = rewards[max(0, (t - window)) : (t + 1)].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


def watch_agent(env: gym.Env, policyModel: Callable):
    """Function to watch the trained agent play."""

    s, info = env.reset()
    done, truncated = False, False

    total_rewards = 0

    while not (done or truncated):
        a = np.argmax(policyModel.predict(s)[0])
        s_next, r, done, truncated, info = env.step(a)

        total_rewards += r

        s = s_next

    print(f"Total Rewards: {total_rewards}")


class FeatureTransformers:
    def __init__(self, n_components: int = 500):
        self.rbf_featurizer = FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components)),
            ]
        )

        self.scaler = StandardScaler()

        self.dimension = None

    def fit(self, X: np.ndarray):
        X = self.scaler.fit_transform(X)
        self.rbf_featurizer.fit(X)

        self.dimension = self.rbf_featurizer.transform(X).shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X)
        return self.rbf_featurizer.transform(X)
