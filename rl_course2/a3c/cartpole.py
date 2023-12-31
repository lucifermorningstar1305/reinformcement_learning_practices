"""
Implementation of the A3C algorithm for the CartPole environment.

Citation: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py

"""

from typing import List, Tuple, Dict, Any, Callable, Union

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import torch.multiprocessing as tm
import os
import argparse


from gymnasium.wrappers.record_video import RecordVideo
from utils import watch_agent

N_GAMES = 3000
T_MAX = 5


class SharedAdam(torch.optim.Adam):
    def __init__(
        self,
        params: List,
        lr: float = 1e-3,
        betas: Tuple = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]

                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self, D: int, K: int, gamma: float = 0.99, reg: float = 0.3):
        super().__init__()

        self.gamma = gamma
        self.reg = reg

        self.policy_network = nn.Sequential(
            nn.Linear(in_features=D, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=K),
        )

        self.value_network = nn.Sequential(
            nn.Linear(in_features=D, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )

        self.states = list()
        self.actions = list()
        self.rewards = list()

    def forward(self, X: torch.Tensor) -> Tuple:
        q_val = self.policy_network(X)
        v_val = self.value_network(X)

        return q_val, v_val

    def remember(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def calc_return(self, done: bool) -> torch.Tensor:
        states = torch.tensor(self.states, dtype=torch.float)
        _, v_val = self.forward(states)

        R = v_val[-1] * (1 - int(done))

        batch_return = list()

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            batch_return.append(R)

        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return

    def calc_loss(self, done: bool) -> float:
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)

        returns = self.calc_return(done)

        q_val, v_val = self.forward(states)
        v_val = v_val.reshape(-1)
        critic_loss = (returns - v_val) ** 2

        probs = F.softmax(q_val, dim=-1)
        dist = tdist.Categorical(probs=probs)
        log_probs = dist.log_prob(actions)
        entropy = -torch.sum(probs * torch.log(probs), axis=1)
        actor_loss = -(log_probs * (returns - v_val) + self.reg * entropy)

        loss = (actor_loss + critic_loss).mean()

        return loss

    def sample_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor([state], dtype=torch.float)

        q_val, _ = self.forward(state)
        probs = F.softmax(q_val, dim=-1)
        dist = tdist.Categorical(probs=probs)

        action = dist.sample().numpy()[0]

        return action

    def save(self, path1: str, path2: str):
        torch.save(self.policy_network.state_dict(), path1)
        torch.save(self.value_network.state_dict(), path2)


class Agent(tm.Process):
    def __init__(
        self,
        env: gym.Env,
        global_actor_critic_model: ActorCritic,
        optimizer: SharedAdam,
        D: int,
        K: int,
        worker_num: int,
        global_ep_idx: tm.Value,
        gamma: float = 0.99,
        reg: float = 0.3,
    ):
        super(Agent, self).__init__()

        self.env = env
        self.global_actor_critic_model = global_actor_critic_model
        self.local_actor_critic_model = ActorCritic(D=D, K=K, gamma=gamma, reg=reg)
        self.name = f"w{worker_num:02d}"
        self.global_ep_idx = global_ep_idx
        self.optimizer = optimizer

    def run(self):
        t_step = 1

        while self.global_ep_idx.value < N_GAMES:
            obs, info = self.env.reset()
            done, truncated = False, False

            score = 0

            self.local_actor_critic_model.clear_memory()

            while not (done or truncated):
                action = self.local_actor_critic_model.sample_action(state=obs)

                next_obs, r, done, truncated, info = self.env.step(action)

                score += r

                self.local_actor_critic_model.remember(
                    state=obs, action=action, reward=r
                )

                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic_model.calc_loss(done=done)
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Copy params from local model to global model

                    for lparams, gparams in zip(
                        self.local_actor_critic_model.parameters(),
                        self.global_actor_critic_model.parameters(),
                    ):
                        gparams._grad = lparams.grad

                    self.optimizer.step()

                    self.local_actor_critic_model.load_state_dict(
                        self.global_actor_critic_model.state_dict()
                    )

                    self.local_actor_critic_model.clear_memory()

                t_step += 1
                observation = next_obs

            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1

            print(
                f"Worker: {self.name}, Episode: {self.global_ep_idx.value}, Reward: {score:.2f}"
            )


if __name__ == "__main__":
    if not os.path.exists("./models"):
        os.mkdir("./models")

    env = gym.make("CartPole-v1", max_episode_steps=2000)
    D = env.observation_space.sample().shape[0]
    K = env.action_space.n

    global_actor_critic_model = ActorCritic(D=D, K=K, gamma=0.99, reg=0.01)
    global_actor_critic_model.share_memory()

    optimizer = SharedAdam(
        global_actor_critic_model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.99,
    )

    global_ep = tm.Value("i", 0)

    workers = [
        Agent(
            env=env,
            global_actor_critic_model=global_actor_critic_model,
            optimizer=optimizer,
            D=D,
            K=K,
            worker_num=i,
            global_ep_idx=global_ep,
            gamma=0.99,
            reg=0.01,
        )
        for i in range(tm.cpu_count())
    ]

    [w.start() for w in workers]
    [w.join() for w in workers]

    global_actor_critic_model.save(
        "./models/cart_pole_policy_network.pt", "./models/cart_pole_value_network.pt"
    )
