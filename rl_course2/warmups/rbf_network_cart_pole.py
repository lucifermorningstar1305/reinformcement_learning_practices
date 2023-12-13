"""
An implementation of the RBF Neural Network for Cart Pole.
"""

from typing import Any, Dict, List, Tuple, Callable, Union

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from gymnasium.wrappers.record_video import RecordVideo


GAMMA = 0.9
ALPHA = 1e-4
LR_TYPE = "constant"


def epsilon_greedy(model: Callable, state: np.ndarray, eps: float = 0.1):
    p = np.random.random()

    if p < eps:
        return model.env.action_space.sample()

    else:
        return np.argmax(model.predict(state))


def gather_samples(env: gym.Env, n_episodes: int) -> List:
    """Function to gather samples by playing random episodes"""

    samples = list()

    for _ in range(n_episodes):
        s, info = env.reset()
        done, truncated = False, False

        while not (done or truncated):
            a = env.action_space.sample()
            s_next, r, done, truncated, info = env.step(a)

            samples.append(s)

            s = s_next

    return samples


class FeatureTransformer:
    def __init__(self, n_components: int):
        self.rbf_featurizer = FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components)),
            ]
        )

        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray):
        X = self.scaler.fit_transform(X)
        self.rbf_featurizer.fit(X)

    def transform(self, X: Union[np.ndarray, List]) -> np.ndarray:
        X = self.scaler.transform([X])
        return self.rbf_featurizer.transform(X)


class Model:
    def __init__(self, env: gym.Env, n_episodes: int = 10_000, n_components: int = 500):
        samples = gather_samples(env=env, n_episodes=n_episodes)

        self.env = env

        self.feature_transformer = FeatureTransformer(n_components=n_components)
        self.feature_transformer.fit(np.asarray(samples))

        self.models = list()

        for _ in range(self.env.action_space.n):
            s, info = env.reset()
            model = SGDRegressor(alpha=ALPHA, learning_rate=LR_TYPE)
            model.partial_fit(self.feature_transformer.transform(s), [0])
            self.models.append(model)

    def predict(self, state: np.ndarray) -> np.ndarray:
        X = self.feature_transformer.transform(state)

        assert (
            len(X.shape) == 2
        ), f"Expected transformed feature to be a 2D array. Found {X.shape}"

        results = [m.predict(X) for m in self.models]

        results = np.stack(results).T

        assert (
            len(results.shape) == 2
        ), f"Expected prediction results to be a 2D array. Found {results.shape}"

        return results

    def update(self, state: np.ndarray, action: int, G: float):
        X = self.feature_transformer.transform(state)
        assert (
            len(X.shape) == 2
        ), f"Expected transformed feature to be a 2D array. Found {X.shape}"

        self.models[action].partial_fit(X, [G])


def test_agent(model: Callable, env: gym.Env, n_episodes: int = 20) -> float:
    """Function to test the agent using greedy-policy."""

    reward_per_episode = list()

    for _ in range(n_episodes):
        s, info = env.reset()
        done, truncated = False, False
        episode_reward = 0

        while not (done or truncated):
            a = epsilon_greedy(model=model, state=s, eps=0)
            s_next, r, done, truncated, info = env.step(a)
            episode_reward += r
            s = s_next

        reward_per_episode.append(episode_reward)

    return np.mean(reward_per_episode)


def watch_agent(model: Callable, env: gym.Env):
    """Function to watch the agent play."""
    s, info = env.reset()
    done, truncated = False, False
    total_reward = 0

    while not (done or truncated):
        a = epsilon_greedy(model=model, state=s, eps=0)
        s_next, r, done, truncated, info = env.step(a)

        total_reward += r
        s = s_next

    print(f"Total Reward earned: {total_reward}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    model = Model(env=env, n_episodes=1500)
    watch_agent(model=model, env=env)

    rewards_per_episode = list()
    n_episodes = 2000

    for episode in range(n_episodes):
        s, info = env.reset()
        done, truncated = False, False
        episode_reward = 0

        while not (done or truncated):
            a = epsilon_greedy(model=model, state=s, eps=0.1)
            s_next, r, done, truncated, info = env.step(a)

            max_a = np.max(model.predict(s_next)[0])
            G = r + GAMMA * max_a

            model.update(state=s, action=a, G=G)

            episode_reward += r

            s = s_next

        rewards_per_episode.append(episode_reward)

        if episode % 100 == 0:
            print(
                f"Episode: {episode} | Average Reward: {np.mean(rewards_per_episode):.3f}"
            )

    plt.plot(rewards_per_episode)
    plt.title("Rewards per episode")
    plt.show()

    running_avg = list()

    for i in range(len(rewards_per_episode)):
        window = rewards_per_episode[i : i + 100]
        running_avg.append(np.mean(running_avg))

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

    test_reward = test_agent(model=model, env=env)
    print(f"Average Test Reward: {test_reward:.5f}")

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env=env, video_folder="./video", name_prefix="cart_pole")
    env.reset()
    env.start_video_recorder()
    watch_agent(model=model, env=env)
