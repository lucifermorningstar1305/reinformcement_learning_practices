"""
An implementation of the policy gradients algorithm for the Cart Pole environment.
"""

from typing import List, Dict, Tuple, Any, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import os
import argparse

from gymnasium.wrappers.record_video import RecordVideo
from utils import plot_running_avg, watch_agent

GAMMA = 0.99


class SimpleNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: List,
        n_actions: int,
        act_fn: Callable = nn.Tanh,
        use_bias: bool = True,
    ):
        super().__init__()

        m1 = input_dim

        self.layers = nn.Sequential()

        for idx, m2 in enumerate(hidden_layer_sizes):
            self.layers.add_module(
                f"layer_{idx+1}",
                nn.Linear(in_features=m1, out_features=m2, bias=use_bias),
            )

            self.layers.add_module(f"activation_{idx+1}", act_fn())
            m1 = m2

        self.layers.add_module(
            f"last_layer",
            nn.Linear(in_features=m1, out_features=n_actions, bias=True),
        )

        if n_actions > 1:
            self.layers.add_module(f"softmax", nn.Softmax(dim=-1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)


class PolicyModel:
    def __init__(
        self,
        D: int,
        K: int,
        hidden_layer_sizes: List,
        lr: float = 1e-3,
        device: str = "cuda",
        optimizer: str = "sgd",
    ):
        if optimizer not in ["sgd", "adam", "adagrad", "rmsprop"]:
            raise Exception(
                f"Expected optimizer to be either sgd/adam/adagrad/rmsprop. Found {optimizer}"
            )

        opt_dict = dict(
            sgd=torch.optim.SGD,
            adam=torch.optim.Adam,
            adagrad=torch.optim.Adagrad,
            rmsprop=torch.optim.RMSprop,
        )

        self.n_actions = K
        self.device = device
        self.simple_model = SimpleNNModel(
            input_dim=D, hidden_layer_sizes=hidden_layer_sizes, n_actions=K
        ).to(device=device)

        self.optimizer = opt_dict[optimizer](
            params=self.simple_model.parameters(), lr=lr
        )

    def partial_fit(self, X: np.ndarray, actions: np.ndarray, advantages: np.ndarray):
        X = np.atleast_2d(X)
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)

        X = torch.from_numpy(X).float().to(device=self.device)
        actions = torch.from_numpy(actions).long().to(device=self.device)
        advantages = torch.from_numpy(advantages).float().to(device=self.device)

        self.optimizer.zero_grad()

        p_a_given_s = self.simple_model(X)
        selected_probs = torch.log(
            torch.sum(
                p_a_given_s
                * F.one_hot(actions, num_classes=self.n_actions).to(device=self.device),
                dim=1,
            )
        )
        cost = -torch.sum(advantages * selected_probs)
        cost.backward()
        self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X = torch.from_numpy(X).float().to(device=self.device)
        pred = self.simple_model(X).detach().cpu().numpy()
        return pred

    def sample_action(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X = torch.from_numpy(X).float().to(device=self.device)
        p = self.simple_model(X).detach().cpu().numpy()[0]

        return np.random.choice(len(p), p=p)


class ValueModel:
    def __init__(
        self,
        D: int,
        hidden_layer_sizes: List,
        lr: float = 1e-3,
        device: str = "cuda",
        optimizer: str = "sgd",
    ):
        if optimizer not in ["sgd", "adam", "adagrad", "rmsprop"]:
            raise Exception(
                f"Expected optimizer to be either sgd/adam/adagrad/rmsprop. Found {optimizer}"
            )

        opt_dict = dict(
            sgd=torch.optim.SGD,
            adam=torch.optim.Adam,
            adagrad=torch.optim.Adagrad,
            rmsprop=torch.optim.RMSprop,
        )

        self.simple_model = SimpleNNModel(
            input_dim=D, hidden_layer_sizes=hidden_layer_sizes, n_actions=1
        ).to(device=device)

        self.optimizer = opt_dict[optimizer](
            params=self.simple_model.parameters(), lr=lr
        )

        self.device = device

    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)

        X = torch.from_numpy(X).float().to(device=self.device)
        y = torch.from_numpy(y).float().to(device=self.device)

        self.optimizer.zero_grad()

        yhat = self.simple_model(X)
        yhat = yhat.reshape(-1)

        cost = torch.sum((y - yhat) ** 2)

        cost.backward()
        self.optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        X = torch.from_numpy(X).float().to(device=self.device)
        pred = self.simple_model(X).detach().cpu().numpy()

        return pred


def play_one_episode_td(
    env: gym.Env, policy_model: Callable, value_model: Callable
) -> float:
    s, info = env.reset()
    done, truncated = False, False
    total_reward = 0

    while not (done or truncated):
        a = policy_model.sample_action(s)
        s_next, r, done, truncated, info = env.step(a)
        V_next = value_model.predict(s_next)
        assert (
            len(V_next.shape) == 2
        ), f"V_next is supposed to be a 2D array. Found {len(V_next.shape)}D array with shape: {V_next.shape}"
        V_next = V_next[0]

        G = r + GAMMA * V_next
        advantage = G - value_model.predict(s)

        policy_model.partial_fit(X=s, actions=a, advantages=advantage)
        value_model.partial_fit(X=s, y=G)

        total_reward += r
        s = s_next

    return total_reward


def play_one_episode_mc(
    env: gym.Env, policy_model: Callable, value_model: Callable
) -> float:
    s, info = env.reset()
    done, truncated = False, False
    total_reward = 0

    states, actions, rewards = list(), list(), list()
    r = 0
    while not (done or truncated):
        a = policy_model.sample_action(s)

        states.append(s)
        actions.append(a)
        rewards.append(r)

        s_next, r, done, truncated, info = env.step(a)

        if done:
            r = -200

        if r == 1:
            total_reward += r

        s = s_next

    # Save the final state
    a = policy_model.sample_action(s)
    states.append(s)
    actions.append(a)
    rewards.append(r)

    returns = list()
    advantages = list()

    G = 0
    for s, r in zip(reversed(states), reversed(rewards)):
        returns.append(G)
        advantages.append(G - value_model.predict(s)[0])
        G = r + GAMMA * G

    returns.reverse()
    advantages.reverse()

    policy_model.partial_fit(states, actions, advantages)
    value_model.partial_fit(states, returns)

    return total_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_video",
        "-v",
        required=False,
        type=int,
        default=0,
        help="whether to save a video of the agent playing or not.",
    )

    parser.add_argument(
        "--video_folder",
        "-F",
        required=False,
        type=str,
        default="./video",
        help="where to save the video file.",
    )

    parser.add_argument(
        "--video_name",
        "-f",
        required=False,
        type=str,
        default="sample_video",
        help="name of the video file.",
    )

    args = parser.parse_args()

    save_video = bool(args.save_video)
    video_folder = args.video_folder
    video_name = args.video_name

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    D = env.observation_space.shape[0]
    K = env.action_space.n

    policy_model = PolicyModel(
        D=D, K=K, hidden_layer_sizes=[], lr=1e-1, optimizer="adagrad"
    )
    value_model = ValueModel(D=D, hidden_layer_sizes=[10], lr=1e-4, optimizer="sgd")

    N = 1000
    total_rewards = np.empty(N)

    for n in range(N):
        total_reward = play_one_episode_mc(
            env=env, policy_model=policy_model, value_model=value_model
        )

        total_rewards[n] = total_reward

        if n % 100 == 0:
            print(
                f"Episode: {n} | Total Reward : {total_reward} | Average Reward (last 100): {total_rewards[max(0, n-100):(n+1)].mean()}"
            )

    print(f"Average reward for last 100 episodes: {total_rewards[-100:].mean()}")

    plt.plot(total_rewards)
    plt.title("Rewards per episode")
    plt.show()

    plot_running_avg(rewards=total_rewards)

    if save_video:
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = RecordVideo(env=env, video_folder=video_folder, name_prefix=video_name)
        env.reset()
        env.start_video_recorder()
    else:
        env = gym.make("CartPole-v1", render_mode="human")

    watch_agent(env=env, policyModel=policy_model)

    # X = np.array([2.34, -3.4, -4.2, -3.6])
    # policy_model = PolicyModel(D=X.shape[0], K=3, hidden_layer_sizes=[32, 64, 128])
    # policy_model.partial_fit(X, 1.0, 0.2)

    # print(policy_model.sample_action(X))

    # value_model = ValueModel(D=X.shape[0], hidden_layer_sizes=[32, 64, 128])
    # value_model.partial_fit(X, -3.4)
