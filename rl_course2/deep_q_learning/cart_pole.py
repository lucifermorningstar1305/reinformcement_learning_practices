"""
An implementation of the Deep Q Network for the CartPole Environment.
"""

from typing import List, Callable, Dict, Tuple, Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym
import os


from utils import plot_running_avg, watch_agent
from gymnasium.wrappers.record_video import RecordVideo


global_iters = 0


class NeuralModel(nn.Module):
    def __init__(
        self,
        D: int,
        K: int,
        hidden_layers: List,
        act_fn: Callable = nn.Tanh,
        use_bias: bool = True,
        final_act_fn: Callable = nn.Identity,
    ):
        super().__init__()

        M1 = D
        self.network = nn.Sequential()

        for idx, M2 in enumerate(hidden_layers):
            self.network.add_module(
                f"layer_{idx}",
                nn.Linear(in_features=M1, out_features=M2, bias=use_bias),
            )

            self.network.add_module(f"activation_{idx}", act_fn())

            M1 = M2

        # Add the final layer
        self.network.add_module(
            f"final_layer", nn.Linear(in_features=M1, out_features=K, bias=use_bias)
        )

        self.network.add_module(f"final_activation", final_act_fn())

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


class DQN:
    def __init__(
        self,
        D: int,
        K: int,
        hidden_layers: List,
        gamma: float = 0.99,
        max_experiences: int = 10_000,
        min_experiences: int = 100,
        batch_sz: int = 32,
        device: str = "cuda",
        optimizer: str = "adam",
        lr: float = 1e-3,
    ):
        """
        An implementation of the DQN Network.

        Parameters:
        :param D: represents the dimension of the input.
        :param K: represents the number of actions.
        :param hidden_layers: represents the number of hidden neurons at each layer.
        :param gamma: the GAMMA value.
        :param max_experiences: represents the Replay-Buffer size.
        :param min_experiences: represents the minimum number of experiences to collect for training.
        :param batch_sz: represents the batch size
        :param device: represents the device to use for training.
        :param optimizer: name of the optimizer to be used for training.
        :param lr: the learning rate
        """

        self.D = D
        self.K = K
        self.hidden_layers = hidden_layers
        self.gamma = gamma
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.device = device

        self.model = NeuralModel(
            D=self.D, K=self.K, hidden_layers=self.hidden_layers
        ).to(device=self.device)

        self.experience = dict(s=list(), a=list(), r=list(), s_next=list(), done=list())

        optimizers = dict(
            sgd=torch.optim.SGD,
            adam=torch.optim.Adam,
            adagrad=torch.optim.Adagrad,
            rmsprop=torch.optim.RMSprop,
        )

        self.optimizer = optimizers[optimizer](self.model.parameters(), lr=lr)

    def train(self, target_network: Any):
        if len(self.experience["s"]) < self.min_experiences:
            return

        batch_idx = np.random.choice(
            len(self.experience["s"]), size=self.batch_sz, replace=False
        )

        states = np.array([self.experience["s"][i] for i in batch_idx])
        actions = np.array([self.experience["a"][i] for i in batch_idx])
        rewards = np.array([self.experience["r"][i] for i in batch_idx])
        next_states = np.array([self.experience["s_next"][i] for i in batch_idx])
        dones = np.array([self.experience["done"][i] for i in batch_idx])

        states = torch.from_numpy(states).float().to(device=self.device)
        actions = torch.from_numpy(actions).long().to(device=self.device)
        rewards = torch.from_numpy(rewards).float().to(device=self.device)
        next_states = torch.from_numpy(next_states).float().to(device=self.device)

        self.optimizer.zero_grad()
        yhat = self.model(states)
        selected_actions_values = torch.sum(
            yhat * F.one_hot(actions, num_classes=self.K).to(device=self.device), dim=1
        )
        next_Q, _ = torch.max(target_network.predict(next_states), axis=1)
        targets = torch.tensor(
            [
                r + self.gamma * next_q if not done else r
                for r, next_q, done in zip(rewards, next_Q, dones)
            ]
        ).to(device=self.device)

        cost = torch.sum((targets - selected_actions_values) ** 2)
        cost.backward()
        self.optimizer.step()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(X, np.ndarray):
            X = np.atleast_2d(X)
            X = torch.from_numpy(X).float().to(device=self.device)
        yhat = self.model(X)
        return yhat

    def copy_from(self, other):
        cur_param = self.model.parameters()
        other_params = other.model.parameters()

        for p, q in zip(cur_param, other_params):
            p.data.copy_(q.data)

    def add_experience(
        self,
        s: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        s_next: np.ndarray,
        done: bool,
    ):
        if len(self.experience["s"]) >= self.max_experiences:
            self.experience["s"].pop(0)
            self.experience["a"].pop(0)
            self.experience["r"].pop(0)
            self.experience["s_next"].pop(0)
            self.experience["done"].pop(0)

        self.experience["s"].append(s)
        self.experience["a"].append(a)
        self.experience["r"].append(r)
        self.experience["s_next"].append(s_next)
        self.experience["done"].append(done)

    def sample_action(self, s: np.ndarray, eps: float) -> np.ndarray:
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict(s).detach().cpu().numpy()[0])


def play_one_episode(
    env: gym.Env,
    model: DQN,
    tmodel: DQN,
    eps: float,
    gamma: float,
    copy_period: int = 100,
) -> float:
    """Function to play one episode of the environment and return the episode's return."""

    global global_iters

    s, info = env.reset()
    done, truncated = False, False
    total_reward = 0

    while not (done or truncated):
        a = model.sample_action(s=s, eps=eps)
        s_next, r, done, truncated, info = env.step(a)

        total_reward += r

        if done:
            r = -200

        model.add_experience(s=s, a=a, r=r, s_next=s_next, done=done)
        model.train(tmodel)

        global_iters += 1

        if global_iters % copy_period == 0:
            tmodel.copy_from(model)

        s = s_next

    return total_reward


if __name__ == "__main__":
    env = gym.make("CartPole-v1", max_episode_steps=2000, render_mode="rgb_array")
    gamma = 0.99
    copy_period = 50

    D = len(env.observation_space.sample())
    K = env.action_space.n

    hidden_layers = [200, 200]

    model = DQN(D=D, K=K, hidden_layers=hidden_layers, gamma=gamma, lr=1e-2)
    tmodel = DQN(D=D, K=K, hidden_layers=hidden_layers, gamma=gamma, lr=1e-2)

    N = 500
    total_rewards = np.empty(N)
    costs = np.empty(N)

    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        total_reward = play_one_episode(
            env=env,
            model=model,
            tmodel=tmodel,
            eps=eps,
            gamma=gamma,
            copy_period=copy_period,
        )

        total_rewards[n] = total_reward

        if n % 100 == 0:
            print(
                f"Episode: {n} | Total Reward: {total_reward:.2f} | Avg Reward (last 100): {total_rewards[max(0, n-100):n+1].mean():.5f}"
            )

    print(f"Average reward for last 100 episode: {total_rewards[-100:].mean():.5f}")

    plt.plot(total_rewards)
    plt.title("Total Reward earned per episode")
    plt.show()

    plot_running_avg(rewards=total_rewards)

    env = gym.make("CartPole-v1", render_mode="human")
    watch_agent(env=env, policyModel=model)
