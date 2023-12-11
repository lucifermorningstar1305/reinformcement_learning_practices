from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import time

from sklearn.kernel_approximation import RBFSampler

GAMMA = 0.9
ALPHA = 0.1
EPS = 0.1


def epsilon_greedy(model: Any, state: Tuple, eps: float = 0.1) -> int:
    p = np.random.random()

    if p < (1 - eps):
        vals = model.predict_all_actions(state)
        return np.argmax(vals)

    else:
        return model.env.action_space.sample()


def gather_samples(env: gym.Env, n_epsiodes: int = 10_000) -> List:
    samples = list()

    for _ in range(n_epsiodes):
        s, info = env.reset()
        done = False

        while not done:
            a = env.action_space.sample()
            sa = np.concatenate((s, [a]))
            samples.append(sa)

            s_next, r, done, truncated, info = env.step(a)
            s = s_next

    return samples


class Model:
    def __init__(self, env: gym.Env, n_episodes: int = 10_000):
        self.env = env
        samples = gather_samples(env=env, n_epsiodes=n_episodes)

        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)

        dims = self.featurizer.n_components

        self.w = np.zeros(dims)

    def predict(self, state: Tuple, action: int) -> np.ndarray:
        sa = np.concatenate((state, [action]))
        x = self.featurizer.transform([sa])[0]

        return x @ self.w

    def predict_all_actions(self, state: Tuple) -> List:
        return [self.predict(state, a) for a in range(self.env.action_space.n)]

    def grad(self, state: Tuple, action: int) -> np.ndarray:
        sa = np.concatenate((state, [action]))
        x = self.featurizer.transform([sa])[0]

        return x


def test_agent(model: Model, env: gym.Env, n_episodes: int = 20) -> float:
    reward_per_episode = list()

    for episode in range(n_episodes):
        s, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        while not (done or truncated):
            a = epsilon_greedy(model=model, state=s, eps=0)
            s_next, r, done, truncated, info = env.step(a)
            episode_reward += r
            s = s_next
        reward_per_episode.append(episode_reward)

    return np.mean(reward_per_episode)


def watch_agent(model: Model, env: gym.Env, eps: float):
    done = False
    truncated = False
    s, info = env.reset()
    episode_reward = 0

    while not (done or truncated):
        a = epsilon_greedy(model=model, state=s, eps=eps)
        s_next, r, done, truncated, info = env.step(a)
        episode_reward += r
        s = s_next

    print(f"Episode Reward: {episode_reward}")


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    model = Model(env=env)

    reward_per_episode = list()

    watch_agent(model=model, env=env, eps=0)

    n_episodes = 1500

    for episode in range(n_episodes):
        s, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            a = epsilon_greedy(model=model, state=s, eps=EPS)
            s_next, r, done, truncated, info = env.step(a)

            if done:
                y = r

            else:
                val = model.predict_all_actions(state=s_next)
                y = r + GAMMA * np.max(val)

            err = y - model.predict(state=s, action=a)
            grad = model.grad(state=s, action=a)
            model.w += ALPHA * err * grad
            s = s_next

            episode_reward += r

        reward_per_episode.append(episode_reward)

        if episode % 100 == 0:
            print(f"Episode: {episode} | Total reward: {episode_reward}")

    plt.plot(reward_per_episode)
    plt.title("Reward per Episode")
    plt.show()

    test_reward = test_agent(model=model, env=env)
    print(f"Average test reward: {test_reward:.2f}")

    env = gym.make("CartPole-v1", render_mode="human")
    watch_agent(model=model, env=env, eps=0)
