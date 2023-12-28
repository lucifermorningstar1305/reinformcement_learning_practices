"""
An implementation of the DQN network for the Atari Breakout-v0 game.
"""

from typing import List, Tuple, Any, Dict, Optional, Callable, Union

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import albumentations as A
import cv2
import random
import os
import argparse

from utils import play_atari_game
from gymnasium.wrappers.record_video import RecordVideo
from datetime import datetime


MAX_EXPERIENCES = 500_000
MIN_EXPERIENCES = 50_000
TARGET_UPDATE_PERIOD = 10_000
IM_SIZE = 84
K = 4


class ImageTransform:
    def __init__(self):
        self.compose = A.Compose(
            [
                A.Crop(x_min=0, y_min=34, x_max=160, y_max=200, always_apply=True),
                A.Resize(
                    height=IM_SIZE,
                    width=IM_SIZE,
                    interpolation=cv2.INTER_NEAREST,
                    always_apply=True,
                ),
            ]
        )

    def transform(self, img: np.ndarray) -> np.ndarray:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_tf = self.compose(image=gray_img)
        return img_tf["image"]


def update_state(state: np.ndarray, obs_small: np.ndarray) -> np.ndarray:
    return np.append(state[:, :, 1:], np.expand_dims(obs_small, 2), axis=2)


class ReplayMemory:
    def __init__(
        self,
        max_size: int = MAX_EXPERIENCES,
        frame_height: int = IM_SIZE,
        frame_width: int = IM_SIZE,
        agent_history_length: int = 4,
        batch_size: int = 32,
    ):
        """
        An implementation of Replay Memory Buffer for the DQN algorithm

        Parameters:
        -----------
        :param max_size: the maximum number of stored transitions.
        :param frame_height: the height of the frame on the Atari Game.
        :param frame_width: the width of the frame on the Atari Game.
        :param agent_history_length: the number of frames stacked together to makeup a STATE
        :param batch_size: the batch size.
        """

        self.max_size = max_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size

        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.max_size, dtype=np.int32)
        self.rewards = np.empty(self.max_size, dtype=np.float32)
        self.frames = np.empty(
            (self.max_size, self.frame_height, self.frame_width), dtype=np.float32
        )

        self.terminal_flags = np.empty(self.max_size, dtype=bool)

        # Pre-allocate memory for the states and new_states in mini-batch

        self.states = np.empty(
            (
                self.batch_size,
                self.agent_history_length,
                self.frame_height,
                self.frame_width,
            ),
            dtype=np.uint8,
        )

        self.next_states = np.empty(
            (
                self.batch_size,
                self.agent_history_length,
                self.frame_height,
                self.frame_width,
            ),
            dtype=np.uint8,
        )

        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(
        self,
        frame: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal_flag: np.ndarray,
    ):
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError(
                f"Dimensions of the frame is wrong. Expected the frame to be of shape ({self.frame_height}, {self.frame_width}). Found {frame.shape}"
            )

        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal_flag

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.max_size

    def _get_state(self, index: int) -> np.ndarray:
        if self.count == 0:
            raise ValueError("The Replay Memory Buffer is EMPTY!")

        if index < self.agent_history_length - 1:
            raise ValueError(f"Index must be at least {self.agent_history_length - 1}")

        return self.frames[index - self.agent_history_length + 1 : index + 1, ...]

    def _get_valid_indices(self) -> np.ndarray:
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if (
                    index >= self.current
                    and index - self.agent_history_length <= self.current
                ):
                    continue
                if self.terminal_flags[index - self.agent_history_length : index].any():
                    continue
                break
            self.indices[i] = index

    def get_minibatch(self) -> Tuple:
        if self.count < self.agent_history_length:
            raise ValueError("Not enough memories to get a minibatch")

        self._get_valid_indices()

        for i, idx in enumerate(self.indices):
            self.states[i] = self._get_state(idx - 1)
            self.next_states[i] = self._get_state(idx)

        return (
            np.transpose(self.states, (0, 2, 3, 1)),  # (BZ, H, W, T)
            self.actions[self.indices],
            self.rewards[self.indices],
            np.transpose(self.next_states, (0, 2, 3, 1)),  # (BZ, H, W, T)
            self.terminal_flags[self.indices],
        )


class CNNModel(nn.Module):
    def __init__(self, K: int, cnn_params: List, fully_connected_params: List):
        super().__init__()

        self.network = nn.Sequential()

        for idx, (out_channels, kernel_size, stride) in enumerate(cnn_params):
            self.network.add_module(
                f"conv2d_{idx}",
                nn.LazyConv2d(
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            )

            self.network.add_module(f"activation_{idx}", nn.ReLU())

        self.network.add_module("flatten", nn.Flatten())

        for idx, out_feats in enumerate(fully_connected_params):
            self.network.add_module(f"fc_{idx}", nn.LazyLinear(out_features=out_feats))
            self.network.add_module(f"fc_activation_{idx}", nn.ReLU())

        self.network.add_module("final_layer", nn.LazyLinear(out_features=K))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


class DQN:
    def __init__(
        self,
        K: int,
        cnn_params: List,
        fully_connected_params: List,
        device: str = "cuda",
        lr: float = 1e-3,
        load_path: str = None,
    ):
        self.K = K
        self.cnn_model = CNNModel(
            K=K,
            cnn_params=cnn_params,
            fully_connected_params=fully_connected_params,
        ).to(device=device)
        self.device = device

        self.load(load_path)

        self.optimizer = torch.optim.Adam(self.cnn_model.parameters(), lr=lr)
        self.criterion = nn.HuberLoss(reduction="sum")
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

    def train(
        self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray
    ) -> float:
        states = np.transpose(states, (0, 3, 1, 2))  # (BZ, T, H, W)

        states = torch.from_numpy(states).float().to(device=self.device)
        actions = torch.from_numpy(actions).long().to(device=self.device)
        targets = torch.from_numpy(targets).float().to(device=self.device)

        states /= 255.0
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast_mode.autocast(enabled=True, dtype=torch.float16):
            pred = self.cnn_model(states)
            selected_action_value = torch.sum(
                pred * F.one_hot(actions, num_classes=self.K), dim=1
            )

            cost = self.criterion(selected_action_value, targets)

        self.scaler.scale(cost).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return cost.item()

    def predict(self, states: np.ndarray) -> torch.Tensor:
        states = np.transpose(states, (0, 3, 1, 2))  # (N, T, H, W)
        states = torch.from_numpy(states).float().to(device=self.device)

        states /= 255.0

        return self.cnn_model(states).detach().cpu()

    def copy_from(self, other: Callable):
        cur_params = self.cnn_model.parameters()
        other_params = other.cnn_model.parameters()

        for p, q in zip(cur_params, other_params):
            p.data.copy_(q.data)

    def sample_action(self, state: np.ndarray, eps: float) -> np.uint8:
        if np.random.random() < eps:
            return np.random.choice(self.K)

        else:
            return np.argmax(self.predict([state]).numpy()[0])

    def save(self, path: str):
        torch.save(self.cnn_model.state_dict(), path)

    def load(self, path: str):
        if path is not None:
            self.cnn_model.load_state_dict(torch.load(path))


def learn(
    model: DQN, tmodel: DQN, exp_replay_buffer: ReplayMemory, gamma: float
) -> float:
    """Function to let the network learn"""

    states, actions, rewards, next_states, dones = exp_replay_buffer.get_minibatch()

    # Q value of next states
    nextQ_val = tmodel.predict(next_states).numpy()
    nextQ = np.amax(nextQ_val, axis=1)

    targets = rewards + np.invert(dones).astype(np.float32) * gamma * nextQ

    loss = model.train(states=states, actions=actions, targets=targets)

    return loss


def play_one_episode(
    env: gym.Env,
    total_t: int,
    model: DQN,
    tmodel: DQN,
    replay_buffer: ReplayMemory,
    img_transformer: ImageTransform,
    gamma: float,
    epsilon: float,
    epsilon_change: float,
    epsilon_min: float,
) -> Dict:
    """Function to play only one episode of the game."""

    t0 = datetime.now()
    obs, info = env.reset()
    obs_small = img_transformer.transform(obs)
    state = np.stack([obs_small] * 4, axis=2)

    loss = None

    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    done, truncated = False, False
    while not (done or truncated):
        if total_t % TARGET_UPDATE_PERIOD == 0:
            tmodel.copy_from(model)
            print(
                f"Copied model parameters to target network. total_t = {total_t}, period = {TARGET_UPDATE_PERIOD}"
            )

        action = model.sample_action(state=state, eps=epsilon)
        obs, r, done, truncated, info = env.step(action)
        obs_small = img_transformer.transform(obs)  # (H, W, C)
        next_state = update_state(state, obs_small)

        episode_reward += r

        replay_buffer.add_experience(
            frame=obs_small, action=action, reward=r, terminal_flag=done
        )

        t1 = datetime.now()
        loss = learn(
            model=model, tmodel=tmodel, exp_replay_buffer=replay_buffer, gamma=gamma
        )

        wandb.log({"loss": loss})

        learning_time = datetime.now() - t1
        total_time_training += learning_time.total_seconds()
        num_steps_in_episode += 1
        total_t += 1

        state = next_state
        epsilon = max(epsilon - epsilon_change, epsilon_min)

    return {
        "total_t": total_t,
        "episode_reward": episode_reward,
        "total_training_time": (datetime.now() - t0),
        "num_steps_in_episode": num_steps_in_episode,
        "time_per_step": total_time_training / num_steps_in_episode,
        "epsilon": epsilon,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action",
        "-a",
        type=str,
        required=True,
        help="whether to train/test the models.",
    )
    parser.add_argument(
        "--model_folder",
        "-mF",
        type=str,
        required=False,
        default="./models",
        help="the folder to store the models.",
    )
    parser.add_argument(
        "--model_name",
        "-mf",
        type=str,
        required=False,
        default="atari_breakout_v0.pt",
        help="the name of the model to save.",
    )

    parser.add_argument(
        "--save_video",
        "-s",
        type=int,
        required=False,
        default=0,
        help="whether to save a video of the gameplay or not.",
    )

    parser.add_argument(
        "--video_folder",
        "-V",
        type=str,
        required=False,
        default="./videos",
        help="where to save the video.",
    )

    parser.add_argument(
        "--video_name",
        "-v",
        type=str,
        required=False,
        default="atari_breakout_v0",
        help="the name of the video file.",
    )

    args = parser.parse_args()

    action = args.action
    model_folder = args.model_folder
    model_name = args.model_name
    save_video = args.save_video
    video_folder = args.video_folder
    video_name = args.video_name

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    cnn_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    fully_connected_params = [512]

    gamma = 0.99
    batch_sz = 32
    num_episodes = 3500
    total_t = 0
    experience_replay_buffer = ReplayMemory(batch_size=batch_sz)
    epsiode_rewards = np.empty(num_episodes)

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 500_000

    load_path = None

    if os.path.exists(os.path.join(model_folder, model_name)):
        load_path = os.path.join(model_folder, model_name)

    model = DQN(
        K=K,
        cnn_params=cnn_params,
        fully_connected_params=fully_connected_params,
        device="cuda",
        lr=1e-5,
        load_path=load_path,
    )
    tmodel = DQN(
        K=K,
        cnn_params=cnn_params,
        fully_connected_params=fully_connected_params,
        device="cuda",
        lr=1e-5,
        load_path=load_path,
    )

    img_transformer = ImageTransform()

    if action == "train":
        wandb.init(project="Atari Breakout-v0 DQN", name="dqn")
        env = gym.make("Breakout-v0", render_mode="rgb_array")

        obs, info = env.reset()
        for i in range(MIN_EXPERIENCES):
            action = np.random.choice(K)
            obs, reward, done, truncated, info = env.step(action)
            obs_small = img_transformer.transform(obs)
            experience_replay_buffer.add_experience(
                frame=obs_small, action=action, reward=reward, terminal_flag=done
            )

            if done or truncated:
                obs, info = env.reset()

        t0 = datetime.now()
        for i in range(num_episodes):
            res = play_one_episode(
                env=env,
                total_t=total_t,
                model=model,
                tmodel=tmodel,
                replay_buffer=experience_replay_buffer,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_change=epsilon_change,
                img_transformer=img_transformer,
                epsilon_min=epsilon_min,
            )

            total_t = res["total_t"]
            episode_reward = res["episode_reward"]
            duration = res["total_training_time"]
            num_steps_in_episode = res["num_steps_in_episode"]
            time_per_step = res["time_per_step"]
            epsilon = res["epsilon"]

            epsiode_rewards[i] = episode_reward
            wandb.log({"episode": i})
            wandb.log({"episode_reward": episode_reward})
            wandb.log({"num_steps_in_episode": num_steps_in_episode})

            last_100_avg = epsiode_rewards[max(0, i - 100) : i + 1].mean()

            wandb.log({"avg_reward_last_100": last_100_avg})

            print(
                f"Episode: {i}, Duration: {duration}, Num steps: {num_steps_in_episode}, Reward: {episode_reward:.2f}, Training time per step: {time_per_step}, Avg Reward (last 100): {last_100_avg:.3f}, epsilon: {epsilon:.3f}"
            )

        print(f"Total Duration: {datetime.now() - t0}")
        model.save(os.path.join(model_folder, model_name))

    else:
        if save_video:
            env = gym.make("Breakout-v0", render_mode="rgb_array")
            env = RecordVideo(
                env=env, video_folder=video_folder, name_prefix=video_name
            )

            env.reset()
            env.start_video_recorder()

        else:
            env = gym.make("Breakout-v0", render_mode="human")

        play_atari_game(env=env, model=model, img_transform=img_transformer)
