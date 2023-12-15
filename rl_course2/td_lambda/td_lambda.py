"""
An implementation of the TD-Lambda Algorithm for the Mountain-Car environment.
"""

from typing import List, Callable, Dict, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from gymnasium.wrappers.record_video import RecordVideo


GAMMA = 0.9999


class RegressorModel:
    def __init__(self, D: int, alpha: float = 1e-4):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.alpha = alpha

    def partial_fit(
        self, X: np.ndarray, y: Union[np.ndarray, List], eligibility: np.ndarray
    ):
        self.w += self.alpha * (y - X.dot(self.w)) * eligibility

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.w)


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
        self.dimensions = 0

    def fit(self, X: np.ndarray):
        X = self.scaler.fit_transform(X)
        self.rbf_featurizer.fit(X)

        self.dimensions = self.rbf_featurizer.transform(X).shape[1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform([X])
        return self.rbf_featurizer.transform(X)


class Model:
    def __init__(
        self,
        env: gym.Env,
        n_components: int = 500,
        alpha: float = 1e-4,
        n_episodes: int = 10_000,
    ):
        self.env = env

        samples = np.array([env.observation_space.sample() for _ in range(n_episodes)])

        self.feature_transform = FeatureTransformer(n_components=n_components)
        self.feature_transform.fit(samples)

        D = self.feature_transform.dimensions

        self.models = list()
        for _ in range(env.action_space.n):
            reg_model = RegressorModel(D=D, alpha=alpha)
            self.models.append(reg_model)

        self.eligibilities = np.zeros((env.action_space.n, D))

    def predict(self, state: np.ndarray) -> np.ndarray:
        X = self.feature_transform.transform(state)
        assert (
            len(X.shape) == 2
        ), f"Expected X to be a 2D array. Found X to be a {len(X.shape)}D array, with shape: {X.shape}"

        vals = [m.predict(X) for m in self.models]
        vals = np.stack(vals).T

        return vals

    def update(self, state: np.ndarray, action: int, G: float, lambda_: float):
        X = self.feature_transform.transform(state)
        assert (
            len(X.shape) == 2
        ), f"Expected X to be a 2D array. Found X to be a {len(X.shape)}D array, with shape: {X.shape}"

        self.eligibilities *= GAMMA * lambda_
        self.eligibilities[action] += X[0]

        self.models[action].partial_fit(
            X=X[0], y=G, eligibility=self.eligibilities[action]
        )


def epsilon_greedy(state: np.ndarray, model: Model, eps: float = 0.1) -> int:
    """Function to greedily choose an action."""

    p = np.random.random()

    if p < eps:
        return model.env.action_space.sample()

    else:
        return np.argmax(model.predict(state=state))


def play_one_episode(
    env: gym.Env, model: Model, lambda_: float = 0.7, eps: float = 0.1
) -> float:
    """Function to play 1 episode and return the total reward earned from the episode."""

    s, info = env.reset()
    done, truncated = False, False

    total_reward = 0

    while not (done or truncated):
        a = epsilon_greedy(state=s, model=model, eps=eps)
        s_next, r, done, truncated, info = env.step(a)

        future_q_val = model.predict(state=s_next)

        assert future_q_val.shape == (
            1,
            env.action_space.n,
        ), f"Expected shape of the action to be {(1, env.action_space.n)}. Found {future_q_val.shape}"

        G = r + GAMMA * np.max(future_q_val[0])

        model.update(state=s, action=a, G=G, lambda_=lambda_)

        total_reward += r

        s = s_next

    return total_reward


def test_agent(env: gym.Env, model: Model, n_episodes: int = 20) -> float:
    """Function to test the agent using the greedy policy and calculate the expected return from the n_episodes played."""

    rewards = list()

    for _ in range(n_episodes):
        s, info = env.reset()
        done, truncated = False, False
        reward = 0

        while not (done or truncated):
            a = epsilon_greedy(state=s, model=model, eps=0)
            s_next, r, done, truncated, info = env.step(a)

            reward += r

            s = s_next

        rewards.append(r)

    return np.mean(rewards)


def watch_agent(env: gym.Env, model: Model):
    s, info = env.reset()
    done, truncated = False, False
    reward = 0

    while not (done or truncated):
        a = epsilon_greedy(state=s, model=model, eps=0.0)
        s_next, r, done, truncated, info = env.step(a)

        reward += r

        s = s_next

    print(f"Total reward earned: {reward}")


def plot_running_avg(rewards: List, window: int = 100):
    N = rewards.shape[0]
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = rewards[max(0, t - window) : (t + 1)].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    model = Model(env=env, n_components=500, alpha=1e-2)

    watch_agent(env, model)

    n_episodes = 300
    rewards_per_episode = np.empty(n_episodes)

    for i in range(n_episodes):
        eps = 0.1 * (0.97**i)
        episode_reward = play_one_episode(env=env, model=model, eps=eps)
        rewards_per_episode[i] = episode_reward

        print(f"Episode: {i} | Total Reward earned in this episode: {episode_reward}")

    plt.plot(rewards_per_episode)
    plt.title("Rewards earned per episode")
    plt.show()

    plot_running_avg(rewards=rewards_per_episode, window=100)

    test_reward = test_agent(env=env, model=model)
    print(f"Average reward earned from testing the agent: {test_reward:.5f}")

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = RecordVideo(
        env=env, video_folder="./video", name_prefix="mountain_car_td_lambda"
    )
    env.reset()
    env.start_video_recorder()
    watch_agent(env=env, model=model)
