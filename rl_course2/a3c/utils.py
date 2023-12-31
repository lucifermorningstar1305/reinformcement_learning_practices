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
        a = np.argmax(policyModel.predict(s).detach().cpu().numpy()[0])
        s_next, r, done, truncated, info = env.step(a)

        total_rewards += r

        s = s_next

    print(f"Total Rewards: {total_rewards}")


def update_state(state: np.ndarray, obs_small: np.ndarray) -> np.ndarray:
    """Function to append the recent state into the state variable and remove the oldest using FIFO."""
    return np.append(state[:, :, 1:], np.expand_dims(obs_small, axis=2), axis=2)


def play_atari_game(env: gym.Env, model: Callable, img_transform: Callable):
    """Function to play the atari game."""

    obs, info = env.reset()
    obs_small = img_transform.transform(obs)
    state = np.stack([obs_small] * 4, axis=2)

    done, truncated = False, False

    episode_reward = 0

    while not (done or truncated):
        action = model.predict(np.expand_dims(state, axis=0)).numpy()
        action = np.argmax(action, axis=1)[0]
        obs, reward, done, truncated, info = env.step(action)
        obs_small = img_transform.transform(obs)

        episode_reward += reward

        next_state = update_state(state=state, obs_small=obs_small)

        state = next_state

    print(f"Total reward earned: {episode_reward}")


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
