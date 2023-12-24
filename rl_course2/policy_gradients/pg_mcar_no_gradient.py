"""
This is an implementation of the Hill Climbing algorithm for the MountainCar-v1 (Continuous Spaces) environment. 
No gradient ascent is performed in this script.
"""

from typing import Callable, Dict, List, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import os
import argparse

from gymnasium.wrappers.record_video import RecordVideo
from utils import watch_agent, plot_running_avg, FeatureTransformers

# torch.manual_seed(32)

GAMMA = 0.99


class NeuralModel(nn.Module):
    def __init__(
        self,
        D: int,
        hidden_layer_sizes: List,
        activation: str = "tanh",
        last_activation: Optional[Callable] = None,
        last_zero: bool = False,
    ):
        super().__init__()
        activations = dict(tanh=nn.Tanh, relu=nn.ReLU, lrelu=nn.LeakyReLU)

        M1 = D
        self.model = nn.Sequential()
        for idx, M2 in enumerate(hidden_layer_sizes):
            self.model.add_module(
                f"layer_{idx+1}", nn.Linear(in_features=M1, out_features=M2)
            )
            self.model.add_module(f"activation_{idx+1}", activations[activation]())
            M1 = M2

        self.model.add_module(
            "last_layer", nn.Linear(in_features=M1, out_features=1, bias=False)
        )

        if last_activation is not None:
            self.model.add_module("last_activation", last_activation())

        self.last_zero = last_zero

        self.initialize_weights()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def initialize_weights(self):
        for name, m in self.named_modules():
            if "last_layer" in name and isinstance(m, nn.Linear) and self.last_zero:
                nn.init.constant_(m.weight, 0.0)


class PolicyModel:
    def __init__(
        self,
        ft: Callable,
        D: int,
        mean_hidden_layer_sizes: List,
        var_hidden_layer_sizes: List,
        smoothing_val: float = 1e-5,
        device: str = "cuda",
    ):
        self.ft = ft
        self.D = D
        self.mean_hidden_layer_sizes = mean_hidden_layer_sizes
        self.var_hidden_layer_sizes = var_hidden_layer_sizes
        self.smooth_val = smoothing_val
        self.device = device

        self.mean_model = NeuralModel(
            D=D, hidden_layer_sizes=mean_hidden_layer_sizes, last_zero=True
        ).to(device=device)

        self.var_model = NeuralModel(
            D=D, hidden_layer_sizes=var_hidden_layer_sizes, last_activation=nn.Softplus
        ).to(device=device)

        self.params = list()

        for param in list(self.mean_model.parameters()) + list(
            self.var_model.parameters()
        ):
            self.params += param

    def predict(self, X: np.ndarray) -> torch.Tensor:
        # X = X.reshape(1, -1)
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        X = torch.tensor(X).float().to(device=self.device)

        mu = self.mean_model(X).reshape(-1)
        std = self.var_model(X).reshape(-1) + self.smooth_val

        # norm = torch.distributions.normal.Normal(loc=mu, scale=std)
        norm = torch.distributions.Normal(loc=mu, scale=std)

        return torch.clip(norm.sample(), -1, 1)

    def sample_action(self, X: np.ndarray) -> torch.Tensor:
        a_val = self.predict(X)[0]
        return a_val

    def copy(self):
        clone = PolicyModel(
            self.ft, self.D, self.mean_hidden_layer_sizes, self.var_hidden_layer_sizes
        )

        clone.copy_from(self)
        return clone

    def copy_from(self, other):
        cur_param = self.params
        other_param = other.params

        for p, q in zip(cur_param, other_param):
            p.data.copy_(q.data)

    def perturb_params(self):
        """Function used for Hill Climbing. Adds noise to the params and generates a new random setting"""

        with torch.no_grad():
            for p in list(self.mean_model.parameters()) + list(
                self.var_model.parameters()
            ):
                if len(p.data.size()):
                    noise = (
                        np.random.randn(*tuple(p.data.size()))
                        / np.sqrt(p.data.size(0))
                        * 5.0
                    )

                if np.random.random() < 0.1:
                    p.copy_(
                        torch.nn.Parameter(
                            torch.tensor(noise).float().to(device=self.device)
                        )
                    )

                else:
                    p.copy_(
                        torch.nn.Parameter(
                            p + torch.tensor(noise).float().to(device=self.device)
                        )
                    )


def play_one_episode(env: gym.Env, policy_model: object):
    """Function to play one episode of the environment and return the reward"""

    s, info = env.reset()
    done, truncated = False, False
    total_reward = 0

    while not (done or truncated):
        a = policy_model.sample_action(s).detach().cpu().numpy()
        s_next, r, done, truncated, info = env.step([a])
        s = s_next
        total_reward += r

    return total_reward


def play_multiple_episodes(
    env: gym.Env, policy_model: object, n_episodes: int = 100, print_iters: bool = False
) -> float:
    """Function to play multiple epsiodes of the environment."""

    rewards = np.empty(n_episodes)
    for i in range(n_episodes):
        r = play_one_episode(env=env, policy_model=policy_model)
        rewards[i] = r

        if print_iters:
            print(f"Episode: {i} | Average Reward: {rewards[:(i+1)].mean():.5f}")

    avg_reward = rewards.mean()
    print(f"Average Reward: {avg_reward:.5f}")

    return avg_reward


def random_search(env: gym.Env, policy_model: object) -> Tuple:
    """Function to randomly search the environment and find the best possible model for the given environment"""

    total_rewards = list()
    best_avg_reward = float("-inf")
    best_policy_model = policy_model

    num_episodes_per_param_test = 3

    for t in range(100):
        tmp_model = best_policy_model.copy()
        tmp_model.perturb_params()

        avg_reward = play_multiple_episodes(
            env=env, policy_model=tmp_model, n_episodes=num_episodes_per_param_test
        )

        total_rewards.append(avg_reward)

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_policy_model = tmp_model

    return total_rewards, best_policy_model


if __name__ == "__main__":
    env = gym.make("MountainCarContinuous-v0", max_episode_steps=2000)

    samples = np.array([env.observation_space.sample() for _ in range(10_000)])
    ft = FeatureTransformers(n_components=100)
    ft.fit(samples)
    D = ft.dimension

    policy_model = PolicyModel(
        ft=ft,
        D=D,
        mean_hidden_layer_sizes=[],
        var_hidden_layer_sizes=[],
        smoothing_val=1e-4,
    )

    total_rewards, policy_model = random_search(env=env, policy_model=policy_model)

    print(f"Max reward: {np.max(total_rewards)}")
