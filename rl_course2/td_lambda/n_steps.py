"""
An implementation of the N-Steps process for the MountainCar-v0 Environment.
"""

from typing import List, Tuple, Dict, Any, Union, Callable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym

from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

from gymnasium.wrappers.record_video import RecordVideo


GAMMA = 0.99


class SGDRegressor:
    def __init__(self, **kwargs):
        self.w = None
        self.lr = 1e-3

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        if self.w is None:
            D = X.shape[1]
            self.w = np.random.randn(D) / np.sqrt(D)

        self.w += self.lr * (y - X.dot(self.w)).dot(X)

    def predict(self, X: np.ndarray):
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

    def fit(self, X: np.ndarray):
        X = self.scaler.fit_transform(X)
        self.rbf_featurizer.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform([X])
        return self.rbf_featurizer.transform(X)


class Model:
    def __init__(self, env: gym.Env, n_episodes: int = 10_000, n_components: int = 500):
        # samples = self.gather_samples(env=env, n_episodes=n_episodes)
        samples = np.array([env.observation_space.sample() for _ in range(n_episodes)])

        self.env = env

        self.feature_transformer = FeatureTransformer(n_components=n_components)

        self.feature_transformer.fit(samples)

        self.models = list()

        for _ in range(env.action_space.n):
            s, info = env.reset()
            model = SGDRegressor()
            model.partial_fit(self.feature_transformer.transform(s), [0])
            self.models.append(model)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Function to predict the state-action values for a given state."""

        X = self.feature_transformer.transform(state)
        assert (
            len(X.shape) == 2
        ), f"Expected X to be an 2-D array. Found to be a {len(X.shape)}-D array with shape {X.shape}"

        res = [m.predict(X) for m in self.models]
        res = np.stack(res).T  # Shape: (1, n_actions)

        return res

    def update(self, state: np.ndarray, action: int, G: float):
        """Function to update a model for a specific action."""

        X = self.feature_transformer.transform(state)

        assert (
            len(X.shape) == 2
        ), f"Expected X to be an 2-D array. Found to be a {len(X.shape)}-D array with shape {X.shape}"

        self.models[action].partial_fit(X, [G])

    def gather_samples(self, env: gym.Env, n_episodes: int) -> List:
        """Function to generate random samples for a given environment."""

        samples = list()

        for _ in range(n_episodes):
            s, info = env.reset()
            done, truncated = False, False

            while not (done or truncated):
                a = env.action_space.sample()
                samples.append(a)

                s_next, r, done, truncated, info = env.step(a)

                s = s_next

        return samples


def epsilon_greedy(model: Model, state: np.ndarray, eps: float = 0.1):
    """Function to greedily select an action for a given state"""

    p = np.random.random()

    if p < eps:
        return model.env.action_space.sample()

    else:
        vals = model.predict(state)
        return np.argmax(vals)


def play_one_episode(model: Model, env: gym.Env, n: int = 5, eps: float = 0.1):
    """Function to play one episode of the environment."""

    states, actions, rewards = list(), list(), list()

    multipliers = np.asarray([GAMMA] * n) ** np.arange(n)

    s, info = env.reset()
    done, truncated = False, False

    episode_reward = 0

    while not (done or truncated):
        a = epsilon_greedy(model=model, state=s, eps=eps)

        states.append(s)
        actions.append(a)

        s_next, r, done, truncated, info = env.step(a)

        rewards.append(r)

        if len(rewards) >= n:
            return_upto_obs = multipliers.dot(
                rewards[-n:]
            )  # R(t+1) + GAMMA * R(t+2) + GAMMA^2 * R(t+3) + ... + GAMMA ^ (n-1) * R(t+n)
            G = return_upto_obs + (GAMMA**n) * np.max(model.predict(s_next)[0])
            model.update(state=states[-n], action=actions[-n], G=G)

        episode_reward += r

        s = s_next

    """
    Because OpenAI Gym cuts of the episode after some number of steps, the agent might not still reach the goal.
    Therefore, some modifications are required to be made so that the agent reaches it goal point. Plus when 
    we exit the above loop, there are some states which have not being updated, and requires to be updated.
    """

    rewards = rewards[-n + 1 :]
    states = states[-n + 1 :]
    actions = actions[-n + 1 :]

    if s[0] >= 0.5:
        while len(rewards) > 0:
            G = multipliers[: len(rewards)].dot(rewards)
            model.update(state=states[0], action=actions[0], G=G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    else:
        while len(rewards) > 0:
            guess_rewards = rewards + [-1] * (n - len(rewards))
            G = multipliers.dot(guess_rewards)
            model.update(state=states[0], action=actions[0], G=G)
            rewards.pop(0)
            states.pop(0)
            actions.pop(0)

    return episode_reward


def test_agent(model: Model, env: gym.Env, n_episodes: int = 20) -> float:
    """Function to test the trained agent"""

    rewards = list()

    for _ in range(n_episodes):
        s, info = env.reset()

        done, truncated = False, False

        episode_reward = 0

        while not (done or truncated):
            a = epsilon_greedy(model=model, state=s, eps=0)
            s_next, r, done, truncated, info = env.step(a)
            episode_reward += r

            s = s_next

        rewards.append(episode_reward)

    return np.mean(rewards)


def watch_agent(model: Model, env: gym.Env):
    """Function to watch the agent play."""

    s, info = env.reset()
    done, truncated = False, False

    total_reward = 0
    while not (done or truncated):
        a = epsilon_greedy(model=model, state=s, eps=0)
        s_next, r, done, truncated, info = env.step(a)
        s = s_next

        total_reward += r

    print(f"Total reward earned: {r}")


def plot_cost_to_go(env: gym.Env, model: Model, num_tiles: int = 20):
    x = np.linspace(
        start=env.observation_space.low[0],
        stop=env.observation_space.high[0],
        num=num_tiles,
    )

    y = np.linspace(
        start=env.observation_space.low[1],
        stop=env.observation_space.high[1],
        num=num_tiles,
    )

    X, Y = np.meshgrid(x, y)

    Z = np.apply_along_axis(lambda t: -np.max(model.predict(t)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Cost-to-Go == -V(s)")
    ax.set_title("Cost-to-Go Function")

    fig.colorbar(surf)
    plt.show()


def plot_running_avg(rewards: List, window: int = 100):
    N = len(rewards)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = rewards[max(0, t - window) : (t + 1)].mean()

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    model = Model(env=env)

    watch_agent(model=model, env=env)

    N = 500
    rewards_per_episode = np.empty(N)

    for n in range(N):
        eps = 0.1 * (0.97**n)
        total_reward = play_one_episode(model=model, env=env, eps=eps)

        rewards_per_episode[n] = total_reward
        print(f"Episode: {n} | Total reward: {total_reward}")

    print(f"Average reward for last 100 episodes: {rewards_per_episode[-100:].mean()}")

    plt.plot(rewards_per_episode)
    plt.title("Rewards per episode")
    plt.show()

    plot_running_avg(rewards_per_episode)
    plot_cost_to_go(env=env, model=model)

    test_reward = test_agent(model=model, env=env)
    print(f"Average reward earned by testing: {test_reward}")

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = RecordVideo(env=env, video_folder="./video", name_prefix="mountain_car")
    env.reset()
    env.start_video_recorder()
    watch_agent(model=model, env=env)
