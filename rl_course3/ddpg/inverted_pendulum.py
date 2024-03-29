from typing import Optional, List, Tuple, Dict, Any, Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import os
import sys
import argparse
import wandb

from gymnasium.wrappers.record_video import RecordVideo

np.random.seed(0)


class OUActionNoise(object):
    def __init__(
        self,
        mu: np.ndarray,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[np.ndarray] = None,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0

        self.reset()

    def __call__(self) -> np.ndarray:
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )

        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return f"OrnsteinUlenbeckActionNoise(mu={self.mu}, sigma={self.sigma})"


class ReplayBuffer(object):
    def __init__(self, obs_dim: Tuple, K: int, max_size: int = int(10e6)):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, obs_dim), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, K), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, obs_dim), dtype=np.float32)
        self.reward_memory = np.zeros((self.mem_size), dtype=np.float32)
        self.done_memory = np.zeros((self.mem_size), dtype=np.float32)

        self.size = 0

    def store_transition(
        self,
        s: np.ndarray,
        a: np.ndarray,
        s_next: np.ndarray,
        r: np.ndarray,
        done: bool,
    ):
        self.state_memory[self.mem_cntr] = s
        self.action_memory[self.mem_cntr] = a
        self.next_state_memory[self.mem_cntr] = s_next
        self.reward_memory[self.mem_cntr] = r
        self.done_memory[self.mem_cntr] = float(done)

        self.mem_cntr = (self.mem_cntr + 1) % self.mem_size
        self.size = min(self.size + 1, self.mem_size)

    def sample_minibatch(self, batch_size: int = 64) -> Dict:
        batch = np.random.randint(0, self.size, size=batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        next_states = self.next_state_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.done_memory[batch]

        return {
            "states": states,
            "actions": actions,
            "next_states": next_states,
            "rewards": rewards,
            "dones": dones,
        }


class ActorNetwork(nn.Module):
    def __init__(
        self,
        input_dims: int,
        n_actions: int,
        hidden_layers: List = [400, 300],
        hidden_act: Callable = nn.ReLU,
    ):
        super().__init__()

        M1 = input_dims

        self.actor = nn.Sequential()

        for idx, M2 in enumerate(hidden_layers):
            self.actor.add_module(f"fc_{idx+1}", nn.Linear(M1, M2))
            # self.actor.add_module(f"bn_{idx+1}", nn.BatchNorm1d(M2))
            self.actor.add_module(f"act_{idx+1}", hidden_act())

            M1 = M2

        self.actor.add_module(f"fc_final_layer", nn.Linear(M1, n_actions))
        self.actor.add_module(f"final_act", nn.Tanh())

        # self._init_parameters()

    def _init_parameters(self):
        for name, param in self.actor.named_parameters():
            if name.startswith("fc_"):
                if "final_" in name:
                    nn.init.uniform_(param.data, a=-3e-3, b=3e-3)
                else:
                    f = 1.0 / np.sqrt(param.data.size(0))
                    nn.init.uniform_(param.data, a=-f, b=f)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.actor(X)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims: int,
        n_actions: int,
        hidden_layers: List = [400, 300],
        hidden_act: Callable = nn.ReLU,
    ):
        super().__init__()

        M1 = input_dims + n_actions

        self.critic = nn.Sequential()

        for idx, M2 in enumerate(hidden_layers):
            self.critic.add_module(f"fc_{idx+1}", nn.Linear(M1, M2))
            # self.critic.add_module(f"bn_{idx+1}", nn.BatchNorm1d(M2))
            self.critic.add_module(f"act_{idx+1}", hidden_act())

            M1 = M2

        self.critic.add_module("fc_final_layer", nn.Linear(M1, 1))

        # self._init_parameters()

    def _init_parameters(self):
        for name, param in self.critic.named_parameters():
            if name.startswith("fc_"):
                if "final_" in name:
                    nn.init.uniform_(param.data, a=-3e-3, b=3e-3)
                else:
                    f = 1.0 / np.sqrt(param.data.size(0))
                    nn.init.uniform_(param.data, a=-f, b=f)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        concat_tensor = torch.concatenate([X, A], dim=-1)
        critic_value = self.critic(concat_tensor)

        return torch.squeeze(critic_value)


class Agent(object):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        tau: float = 0.995,
        replay_mem_max_size: int = int(1e6),
        batch_size: int = 64,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        critic_weight_decay: float = 1e-2,
        hidden_layer_sizes: List = [400, 300],
        hidden_act: Callable = nn.ReLU,
        theta: float = 0.15,
        sigma: float = 0.2,
        max_action_value: float = 1.0,
        save_model_folder: str = "./models",
        save_path: str = "ddpg.pt",
        device: str = "cuda",
        save_video: bool = False,
        logger: Optional[Callable] = None,
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.action_dim = action_dim

        self.replay_memory = ReplayBuffer(
            obs_dim=obs_dim, K=action_dim, max_size=replay_mem_max_size
        )

        self.actor_model = ActorNetwork(
            input_dims=obs_dim,
            n_actions=action_dim,
            hidden_layers=hidden_layer_sizes,
            hidden_act=hidden_act,
        )

        self.critic_model = CriticNetwork(
            input_dims=obs_dim,
            n_actions=action_dim,
            hidden_layers=hidden_layer_sizes,
            hidden_act=hidden_act,
        )

        print(self.actor_model)
        print("\n******************************\n")
        print(self.critic_model)

        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(),
            lr=lr_critic,
            weight_decay=critic_weight_decay,
        )

        self.target_actor_model = ActorNetwork(
            input_dims=obs_dim,
            n_actions=action_dim,
            hidden_layers=hidden_layer_sizes,
            hidden_act=hidden_act,
        )

        self.target_critic_model = CriticNetwork(
            input_dims=obs_dim,
            n_actions=action_dim,
            hidden_layers=hidden_layer_sizes,
            hidden_act=hidden_act,
        )

        self._copy_main_to_target()

        self.noise = OUActionNoise(mu=np.zeros(action_dim), theta=theta, sigma=sigma)

        self.max_action_value = max_action_value

        self.device = device

        self.save_model_folder = save_model_folder
        self.save_path = save_path
        self.save_video = save_video
        self.logger = logger

        self._copy_to_device()

    def _copy_main_to_target(self):
        with torch.no_grad():
            for param_tc, param_c in zip(
                self.target_critic_model.parameters(), self.critic_model.parameters()
            ):
                param_tc.copy_(param_c.data)

            for param_ta, param_a in zip(
                self.target_actor_model.parameters(), self.actor_model.parameters()
            ):
                param_ta.copy_(param_a.data)

    def _copy_to_device(self):
        self.actor_model.to(device=self.device)
        self.critic_model.to(device=self.device)
        self.target_actor_model.to(device=self.device)
        self.target_critic_model.to(device=self.device)

    def choose_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        obs = torch.tensor(state, dtype=torch.float).to(device=self.device)
        obs = obs.reshape(1, -1)
        with torch.no_grad():
            mu_obs = self.actor_model(obs)

        mu_obs = mu_obs.cpu().numpy()
        mu_prime = mu_obs + noise_scale * np.random.randn(self.action_dim)
        # mu_prime = mu_obs + torch.tensor(self.noise(), dtype=torch.float).to(
        #     device=self.device
        # )

        mu_prime = mu_prime.reshape(-1)

        return np.clip(
            mu_prime,
            -self.max_action_value,
            self.max_action_value,
        )

    def remember(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: np.ndarray,
        done: bool,
    ):
        self.replay_memory.store_transition(
            s=state, a=action, s_next=next_state, r=reward, done=done
        )

    def unfreeze_critic(self, freeze: bool):
        for param in self.critic_model.parameters():
            param.requires_grad = freeze

    def learn(self):
        batch_data = self.replay_memory.sample_minibatch(batch_size=self.batch_size)

        state_mb, action_mb, reward_mb, next_state_mb, done_mb = (
            batch_data["states"],
            batch_data["actions"],
            batch_data["rewards"],
            batch_data["next_states"],
            batch_data["dones"],
        )

        state_mb, action_mb, reward_mb, next_state_mb, done_mb = (
            torch.tensor(state_mb, dtype=torch.float).to(device=self.device),
            torch.tensor(action_mb, dtype=torch.float).to(device=self.device),
            torch.tensor(reward_mb, dtype=torch.float).to(device=self.device),
            torch.tensor(next_state_mb, dtype=torch.float).to(device=self.device),
            torch.tensor(done_mb, dtype=torch.float).to(device=self.device),
        )

        # Updating the critic network
        self.unfreeze_critic(True)
        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            target_actions = self.target_actor_model(next_state_mb)
            q_prime = self.target_critic_model(next_state_mb, target_actions)

        target = reward_mb + self.gamma * (1 - done_mb) * q_prime

        critic_val = self.critic_model(state_mb, action_mb)

        critic_loss = ((critic_val - target) ** 2).mean()

        critic_loss.backward()
        self.critic_optimizer.step()

        # Updating the actor network
        self.unfreeze_critic(False)
        self.actor_optimizer.zero_grad()

        mu = self.actor_model(state_mb)

        actor_val = self.critic_model(state_mb, mu)
        actor_loss = -torch.mean(actor_val)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()

        if self.logger is None:
            print(
                f"Critic Loss : {critic_loss.item():.4f} | Actor Loss: {actor_loss.item():.4f}"
            )
        else:
            self.logger.log({"actor_loss": actor_loss.item()})
            self.logger.log({"critic_loss": critic_loss.item()})

    def _update_target_networks(self):
        with torch.no_grad():
            for param_c, param_tc in zip(
                self.critic_model.parameters(), self.target_critic_model.parameters()
            ):
                param_tc.data.mul_(self.tau)
                param_tc.data.add_((1 - self.tau) * param_c.data)

            for param_a, param_ta in zip(
                self.actor_model.parameters(), self.target_actor_model.parameters()
            ):
                param_ta.data.mul_(self.tau)
                param_ta.data.add_((1 - self.tau) * param_a)

    def save_model(self):
        if not os.path.exists(self.save_model_folder):
            os.mkdir(self.save_model_folder)

        state_dicts = {
            "actor_model_state_dict": self.actor_model.state_dict(),
            "critic_model_state_dict": self.critic_model.state_dict(),
            "target_actor_model_state_dict": self.target_actor_model.state_dict(),
            "target_critic_model_state_dict": self.target_critic_model.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }

        torch.save(state_dicts, os.path.join(self.save_model_folder, self.save_path))

    def load_model(self, path: str):
        chkpt = torch.load(path)

        self.actor_model.load_state_dict(chkpt["actor_model_state_dict"])
        self.critic_model.load_state_dict(chkpt["critic_model_state_dict"])
        self.target_actor_model.load_state_dict(chkpt["target_actor_model_state_dict"])
        self.target_critic_model.load_state_dict(
            chkpt["target_critic_model_state_dict"]
        )
        self.actor_optimizer.load_state_dict(chkpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(chkpt["critic_optimizer_state_dict"])

    def test_agent(
        self,
        env_id: str,
        video_folder: Optional[str] = None,
        video_filename: Optional[str] = None,
        n_episodes: int = 5,
    ) -> float:
        if self.save_video:
            if video_folder is None or video_filename is None:
                raise Exception(
                    f"Found one of video_folder or video_filename to be None. Expected a path or name for the video_file"
                )

            else:
                env = gym.make(env_id, render_mode="rgb_array")
                env = RecordVideo(
                    env=env,
                    video_folder=video_folder,
                    name_prefix=video_filename,
                    episode_trigger=lambda x: x % 1 == 0,
                    disable_logger=True,
                )
                env.reset()
                env.start_video_recorder()

        else:
            env = gym.make(env_id, render_mode="human")

        episode_rewards = list()

        self.actor_model.eval()

        with torch.no_grad():
            for i in range(n_episodes):
                s, info = env.reset()
                done, truncated = False, False

                _rewards = 0.0
                while not (done or truncated):
                    a = self.choose_action(s, noise_scale=0.0)
                    s_next, reward, done, truncated, info = env.step(a)
                    _rewards += reward
                    s = s_next

                episode_rewards.append(_rewards)

        env.close()
        episode_rewards = np.array(episode_rewards)
        print(f"Test Return: {episode_rewards.mean():.2f}")

        return episode_rewards.mean()


def plot_running_average(returns: List, window: int = 100):
    N = len(returns)

    running_avg = np.empty(N)

    for i in range(N):
        running_avg[i] = np.mean(returns[max(0, i - window) : i + 1])

    plt.plot(running_avg)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_video",
        type=int,
        required=False,
        default=0,
        help="whether to save the video file or not.",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        required=False,
        default=None,
        help="path to save the video of the agent",
    )

    parser.add_argument(
        "--video_name",
        type=str,
        required=False,
        default=None,
        help="name of the video file to save.",
    )

    parser.add_argument(
        "--training_episodes",
        type=int,
        required=False,
        default=100,
        help="number of training episodes",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        required=False,
        default="./models",
        help="path to save the model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="ddpg.pt",
        help="name of the model file to be saved",
    )
    parser.add_argument(
        "--batch_size", type=int, required=False, default=64, help="the batch size"
    )

    parser.add_argument(
        "--hidden_layer_sizes",
        type=str,
        required=False,
        default="400,300",
        help="the size of the hidden neurons for each layers. Seperated by ,",
    )

    parser.add_argument(
        "--test_interval",
        type=int,
        required=False,
        default=100,
        help="interval at which to test the agent on the environment.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        required=False,
        default=0.99,
        help="the value of the gamma constant",
    )
    parser.add_argument(
        "--tau",
        type=float,
        required=False,
        default=0.995,
        help="the value of the decay rate constant",
    )

    parser.add_argument(
        "--replay_mem_size",
        type=int,
        required=False,
        default=int(10e6),
        help="replay buffer memory size",
    )

    parser.add_argument(
        "--use_wandb",
        type=int,
        required=False,
        default=0,
        help="whether to use wandb for logging",
    )

    parser.add_argument(
        "--start_steps",
        type=int,
        required=False,
        default=10_000,
        help="number of steps to wait before using DDPG choose action (implementation based on OpenAI)",
    )

    parser.add_argument(
        "--steps_per_training_episode",
        type=int,
        required=False,
        default=4000,
        help="number of steps per epoch",
    )

    parser.add_argument(
        "--update_agent_after",
        type=int,
        required=False,
        default=1000,
        help="number of steps after which to update the agent",
    )

    parser.add_argument(
        "--update_agent_every",
        type=int,
        required=False,
        default=50,
        help="update the agent after every n steps.",
    )

    parser.add_argument(
        "--n_test_episodes",
        type=int,
        required=False,
        default=10,
        help="number of episodes to test the agent.",
    )

    parser.add_argument(
        "--save_freq",
        type=int,
        required=False,
        default=1,
        help="number of episodes after which to save the agent.",
    )

    args = parser.parse_args()

    save_video = args.save_video
    video_folder = args.video_folder
    video_name = args.video_name
    n_training_episodes = args.training_episodes
    model_folder = args.model_folder
    model_name = args.model_name
    batch_size = args.batch_size
    hidden_layer_sizes = args.hidden_layer_sizes
    test_interval = args.test_interval
    gamma = args.gamma
    tau = args.tau
    replay_mem_size = args.replay_mem_size
    use_wandb = args.use_wandb
    start_steps = args.start_steps
    steps_per_training_episode = args.steps_per_training_episode
    update_agent_after = args.update_agent_after
    update_agent_every = args.update_agent_every
    n_test_episodes = args.n_test_episodes
    save_freq = args.save_freq

    max_ep_length = 1000

    assert use_wandb in [
        0,
        1,
    ], f"Expected use_wandb to either 0 or 1. Found {use_wandb}"

    assert save_video in [
        0,
        1,
    ], f"Expected save_video to be either 0 or 1. Found save_video to be {save_video}"

    assert (
        gamma > 0 and gamma <= 1
    ), f"Expected gamma to be in the range (0, 1]. Found gamma to be {gamma}"
    assert (
        tau > 0 and tau <= 1
    ), f"Expected tau to be in the range (0, 1]. Found tau to be {tau}"

    hidden_layer_sizes = list(map(lambda x: int(x), hidden_layer_sizes.split(",")))

    logger = None

    if use_wandb:
        wandb.init(project="DDPG", name="inverted_pendulum")
        logger = wandb

    env = gym.make("InvertedPendulum-v4")
    max_action_val = env.action_space.high[0]

    agent = Agent(
        env=env,
        gamma=gamma,
        tau=tau,
        lr_actor=1e-3,
        lr_critic=1e-3,
        critic_weight_decay=0,
        replay_mem_max_size=replay_mem_size,
        batch_size=batch_size,
        hidden_layer_sizes=hidden_layer_sizes,
        max_action_value=max_action_val,
        save_model_folder=model_folder,
        save_path=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_video=save_video,
        logger=logger,
    )

    if os.path.exists(os.path.join(model_folder, model_name)):
        agent.load_model(os.path.join(model_folder, model_name))

    returns = list()
    num_steps = 0

    total_steps = n_training_episodes * steps_per_training_episode

    s, info = env.reset()
    episode_return, episode_length = 0, 0
    done, truncated = False, False

    for i in range(total_steps):
        if i > start_steps:
            a = agent.choose_action(s)
        else:
            a = env.action_space.sample()

        s_next, r, done, truncated, info = env.step(a)

        episode_return += r
        episode_length += 1

        done = False if episode_length == max_ep_length else done

        agent.remember(state=s, action=a, next_state=s_next, reward=r, done=done)

        s = s_next

        if done or truncated:
            returns.append(episode_return)

            s, info = env.reset()
            done, truncated = False, False
            episode_return, episode_length = 0, 0

        if i >= update_agent_after and i % update_agent_every == 0:
            for _ in range(update_agent_every):
                agent.learn()

        if (i + 1) % steps_per_training_episode == 0:
            episode = (i + 1) // steps_per_training_episode

            print(
                f"Episode: {episode} | Episode Return: {returns[-1]:.2f} | Trailing 100 games rewards: {np.mean(returns[-100:]):.4f}"
            )

            if episode % test_interval == 0:
                test_return = agent.test_agent(
                    env_id="InvertedPendulum-v4",
                    video_folder=video_folder,
                    video_filename=video_name,
                    n_episodes=n_test_episodes,
                )

                wandb.log({"test_return": test_return})

            if episode % save_freq == 0:
                agent.save_model()

        # s, info = env.reset()
        # done, truncated = False, False

        # episode_return = 0
        # episode_length = 0

        # while not (done or truncated):
        #     if num_steps > start_steps:
        #         a = agent.choose_action(s)
        #     else:
        #         a = env.action_space.sample()
        #     s_next, r, done, truncated, info = env.step(a)
        #     done = False if episode_length == max_ep_length else done
        #     agent.remember(state=s, action=a, next_state=s_next, reward=r, done=done)

        #     episode_return += r
        #     s = s_next
        #     num_steps += 1
        #     episode_length += 1

        # for _ in range(episode_length):
        #     agent.learn()

        # returns.append(episode_return)

        # if logger is not None:
        #     logger.log({"episode_return": episode_return})
        #     logger.log({"trailing_return(last 100)": np.mean(returns[-100:])})
        # print(
        #     f"Episode: {i} | Episode Return: {episode_return:.4f} | Trailing 100 games average: {np.mean(returns[-100:])}"
        # )
        # if i % test_interval == 0:
        #     agent.test_agent(
        #         env_id="InvertedPendulum-v4",
        #         video_folder=video_folder,
        #         video_filename=f"video_name_episode={i}",
        #         n_episodes=5,
        #     )

        # agent.save_model()

    agent.test_agent(
        env_id="InvertedPendulum-v4",
        video_folder="./test_video/inverted_pendulum",
        video_filename="test_vid",
    )

    plot_running_average(returns=returns, window=100)
