""" 
An implementation of RBF Neural Network for Mountain Car Environment
"""

from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from gymnasium.wrappers.record_video import RecordVideo


GAMMA = 0.99


def epsilon_greedy(model: Any, state: np.ndarray, eps: float = 0.1):
    """A function to greedily select an action using epsilon-greedy algorithm"""

    p = np.random.random()
    if p < eps:
        return model.env.action_space.sample()

    else:
        return np.argmax(model.predict(state))


def gather_samples(env: gym.Env, n_episodes: int = 10_000) -> List:
    """Function to gather samples for an environment."""

    samples = list()
    for _ in range(n_episodes):
        s, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            a = env.action_space.sample()
            # sa = np.concatenate((s, [a]))
            # samples.append(sa)
            samples.append(s)

            s_next, r, done, truncated, info = env.step(a)
            s = s_next

    return samples


class Model:
    def __init__(self, env: gym.Env, n_episodes: int = 10_000, n_components: int = 500):
        samples = gather_samples(env=env, n_episodes=n_episodes)

        self.env = env
        self.featurizer = FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components)),
            ]
        )

        self.scaler = StandardScaler()
        self.scaler.fit(np.asarray(samples))

        self.featurizer.fit(self.scaler.transform(np.asarray(samples)))

        self.models = list()

        for _ in range(env.action_space.n):  # Seperate neurons for each action
            model = SGDRegressor(learning_rate="constant")
            s, info = env.reset()
            model.partial_fit(
                self.featurizer.transform(self.scaler.transform([s])), [0]
            )
            self.models.append(model)

    def predict(self, state: np.ndarray) -> np.ndarray:
        X = self.featurizer.transform(self.scaler.transform([state]))
        res = [m.predict(X) for m in self.models]  # Returns Q(s) -value for each action

        res = np.stack(res, axis=0).T
        # print(res)
        assert (
            len(res.shape) == 2
        ), f"Expected res shape to be a tuple of 2 elements. Found {len(res.shape)}"
        return res

    def update(self, state: np.ndarray, action: int, G: float):
        X = self.featurizer.transform(self.scaler.transform([state]))
        assert (
            len(X.shape) == 2
        ), f"Expected transformed state shape to be a tuple of 2 elements. Found {len(X.shape)}"

        self.models[action].partial_fit(X, [G])


def test_agent(model: Model, env: gym.Env, n_episodes: int = 20) -> float:
    """Function to test the agent using greedy policy."""

    rewards_per_episode = list()

    for _ in range(n_episodes):
        s, info = env.reset()
        done, truncated = False, False
        episode_reward = 0
        while not (done or truncated):
            a = epsilon_greedy(model=model, state=s, eps=0)
            s_next, r, done, truncated, info = env.step(a)
            episode_reward += r

            s = s_next

        rewards_per_episode.append(episode_reward)

    return np.mean(rewards_per_episode)


def watch_agent(model: Model, env: gym.Env):
    """Function to watch the agent play using the greedy policy."""

    episode_reward = 0

    s, info = env.reset()
    done, truncated = False, False

    while not (done or truncated):
        a = epsilon_greedy(model=model, state=s, eps=0)
        s_next, r, done, truncated, info = env.step(a)
        episode_reward += r
        s = s_next

    print(f"Total reward earned: {r:.2f}")


def play_one_episode(model: Model, env: gym.Env, eps: float) -> float:
    """Function to play an episode and collect reward."""

    s, info = env.reset()
    done, truncated = False, False

    episode_reward = 0
    while not (done or truncated):
        a = epsilon_greedy(model=model, state=s, eps=eps)
        s_next, r, done, truncated, info = env.step(a)

        a_preds = model.predict(state=s_next)  # Shape : (1, env.action_space.n)
        max_a = np.max(a_preds[0])

        G = r + GAMMA * max_a
        model.update(state=s, action=a, G=G)

        episode_reward += r

        s = s_next

    return episode_reward, model


if __name__ == "__main__":
    env = gym.make("MountainCar-v0", render_mode="rgb_array")

    model = Model(env=env, n_episodes=1500, n_components=500)

    rewards_per_episode = list()

    watch_agent(model=model, env=env)

    n_episodes = 300

    for episode in range(n_episodes):
        eps = 0.1 * (0.97**episode)
        episode_reward, model = play_one_episode(model=model, env=env, eps=eps)
        rewards_per_episode.append(episode_reward)
        if episode % 100 == 0:
            print(
                f"Episode: {episode} | Average reward earned: {np.mean(rewards_per_episode):.3f}"
            )

    plt.plot(rewards_per_episode)
    plt.title("Rewards per Episode")
    plt.show()

    test_reward = test_agent(model=model, env=env, n_episodes=20)
    print(f"Average test reward earned: {test_reward:.3f}")

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    env = RecordVideo(env=env, video_folder="video", name_prefix="mountain_car")
    env.reset()
    env.start_video_recorder()
    watch_agent(model=model, env=env)
