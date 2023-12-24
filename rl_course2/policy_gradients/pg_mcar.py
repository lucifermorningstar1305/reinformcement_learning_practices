"""
This is an implementation of the Policy Gradient algorithm with value model for the MountainCar Continuous environment.
"""

from typing import List, Tuple, Any, Dict, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from gymnasium.wrappers.record_video import RecordVideo
from utils import watch_agent, plot_running_avg, FeatureTransformers


GAMMA = 0.95


class NeuralModel(nn.Module):
    def __init__(
        self,
        D: int,
        layers: List,
        act_fn: Callable = nn.Tanh,
        bias: bool = True,
        as_policy: bool = True,
    ):
        super().__init__()

        self.network = nn.Sequential()
        self.as_policy = as_policy

        M1 = D
        for idx, M2 in enumerate(layers):
            self.network.add_module(
                f"layer_{idx+1}", nn.Linear(in_features=M1, out_features=M2, bias=bias)
            )
            self.network.add_module(
                f"activation_{idx}",
                act_fn(),
            )

            M1 = M2

        if as_policy:
            self.mean_layer = nn.Linear(in_features=M1, out_features=1, bias=False)
            self.var_layer = nn.Linear(in_features=M1, out_features=1, bias=False)

        else:
            self.last_layer = nn.Linear(in_features=M1, out_features=1, bias=True)

        self.initialize_network()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Z = self.network(X)

        if self.as_policy:
            mu = self.mean_layer(Z)
            std = self.var_layer(Z)

            return mu, F.softplus(std)
        else:
            return self.last_layer(Z)

    def initialize_network(self):
        if self.as_policy:
            for param in self.mean_layer.parameters():
                if isinstance(param, nn.Linear):
                    nn.init.constant_(param.weight, 0.0)


class PolicyModel:
    def __init__(
        self,
        D: int,
        hidden_layers: List,
        ft: Callable,
        smoothing_val: float = 1e-5,
        device: str = "cuda",
        optimizer: str = "adam",
        lr: float = 1e-3,
    ):
        self.model = NeuralModel(
            D=D,
            layers=hidden_layers,
            act_fn=nn.Tanh,
        ).to(device=device)

        self.smoothing_val = smoothing_val
        self.ft = ft
        self.device = device

        optimizers = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
        }
        self.optimizer = optimizers[optimizer](
            self.model.parameters(),
            lr=lr,
        )

    def partial_fit(self, X: np.ndarray, actions: np.ndarray, advantages: np.ndarray):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)

        X = self.ft.transform(X)
        X = torch.from_numpy(X).float().to(device=self.device)
        actions = torch.from_numpy(actions).float().to(device=self.device)
        advantages = torch.from_numpy(advantages).float().to(device=self.device)

        self.optimizer.zero_grad()

        mu, std = self.model(X)
        mu = mu.reshape(-1)
        std = std.reshape(-1) + self.smoothing_val

        norm = torch.distributions.Normal(loc=mu, scale=std)
        log_probs = norm.log_prob(actions)
        cost = -torch.sum(advantages * log_probs + 0.1 * norm.entropy())

        cost.backward()
        self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        X = torch.from_numpy(X).float().to(device=self.device)

        mu, std = self.model(X)
        mu, std = mu.reshape(-1), std.reshape(-1)
        std = std + self.smoothing_val

        norm = torch.distributions.Normal(loc=mu, scale=std)
        pred_a = torch.clip(norm.sample(), -1, 1).detach().cpu().numpy()

        return pred_a

    def sample_action(self, X: np.ndarray) -> np.ndarray:
        pred_a = self.predict(X)[0]
        return pred_a


class ValueModel:
    def __init__(
        self,
        D: int,
        hidden_layers: List,
        ft: Callable,
        device: str = "cuda",
        optimizer: str = "adam",
        lr: float = 1e-3,
    ):
        self.model = NeuralModel(D=D, layers=hidden_layers, as_policy=False).to(
            device=device
        )
        self.ft = ft
        self.device = device

        optimizers = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adagrad": torch.optim.Adagrad,
        }
        self.optimizer = optimizers[optimizer](self.model.parameters(), lr=lr)

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        X = self.ft.transform(X)
        X = torch.from_numpy(X).float().to(device=self.device)
        y = torch.from_numpy(y).float().to(device=self.device)

        self.optimizer.zero_grad()
        yhat = self.model(X)
        yhat = yhat.reshape(-1)
        cost = torch.sum((y - yhat) ** 2)
        cost.backward()
        self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X = self.ft.transform(X)

        X = torch.from_numpy(X).float().to(device=self.device)
        return self.model(X).detach().cpu().numpy()


def play_one_episode(
    env: gym.Env, policy_model: Callable, value_model: Callable
) -> float:
    """Function to play one episode of the enivornment."""

    s, info = env.reset()
    done, truncated = False, False

    total_rewards = 0.0

    while not (done or truncated):
        a = policy_model.sample_action(s)
        s_next, r, done, truncated, info = env.step([a])

        total_rewards += r

        v_next = value_model.predict(s_next)
        G = r + GAMMA * v_next
        advantage = G - value_model.predict(s)

        policy_model.partial_fit(s, a, advantage)
        value_model.partial_fit(s, G)

        s = s_next

    return total_rewards


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0", max_episode_steps=2000)
    samples = np.array([env.observation_space.sample() for _ in range(10_000)])
    ft = FeatureTransformers(n_components=100)
    ft.fit(samples)

    D = ft.dimension
    policy_model = PolicyModel(D=D, hidden_layers=[], ft=ft)
    value_model = ValueModel(D=D, hidden_layers=[], ft=ft, lr=1e-1)

    N = 50
    total_rewards = np.empty(N)

    for n in range(N):
        total_reward = play_one_episode(
            env=env, policy_model=policy_model, value_model=value_model
        )
        total_rewards[n] = total_reward

        print(
            f"Episode: {n}| Total Reward: {total_reward: .2f} | Average Reward (last 100): {total_rewards[max(0, n-100):n+1].mean():.4f}"
        )

    print(
        f"Average reward earned for last 100 episode: {total_rewards[-100:].mean():.3f}"
    )

    plt.plot(total_rewards)
    plt.title("Total Rewards per episode")
    plt.show()

    plot_running_avg(rewards=total_rewards)
