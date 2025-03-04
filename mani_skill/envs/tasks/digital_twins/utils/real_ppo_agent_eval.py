import json
import os
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym

# from examples.baselines.ppo.ppo_rgb import Agent
import numpy as np
import sapien
import torch
import torch.nn as nn
import tyro
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[
                1
            ]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class Agent(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5
        )

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


# ManiSkill specific imports
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode


# TODO (xhin): suppport sim_env arg, makes default value setting much easier
# TODO (xhin): support for RecordEpisode wrapper to log real world evals
@dataclass
class Args:
    real_env_id: str = "RealGrabCube-v1"
    """environment for evalulation"""
    control_freq: int = 5
    control_mode: str = "pd_joint_delta_pos"
    checkpoint: str = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    num_eval_steps: int = 50
    """maximum steps agent can take"""
    debug: bool = False
    """toggle printing state-based obs at every state, single step at a time"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    eval_name: Optional[str] = None
    """name of eval for saving recording"""
    n: int = 1
    """number of evaluations to make"""

    use_sim_obs: bool = False


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cpu")

    # make eval environment
    env_kwargs = {}
    with open(
        "/home/stao/work/research/maniskill/ManiSkill/examples/tutorials/user_kwargs.json",
        "r",
    ) as f:
        user_kwargs = json.load(f)
        # add user_kwargs to env kwargs
        env_kwargs.update(user_kwargs)
    sim_env_test = gym.make(
        "GrabCube-v1",
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        dr=False,
        **env_kwargs
    )
    eval_envs = gym.make(
        args.real_env_id,
        control_freq=args.control_freq,
        control_mode=args.control_mode,
        control_timing=not args.debug,
    )
    sim_env_test = FlattenRGBDObservationWrapper(
        sim_env_test, rgb=True, depth=False, state=True
    )
    eval_envs = FlattenRGBDObservationWrapper(
        eval_envs, rgb=True, depth=False, state=True
    )
    sim_env_test.reset()
    eval_obs, _ = eval_envs.reset()
    sim_obs, _ = sim_env_test.reset()

    # recreate sim ppo_rgb.py trained agent
    agent = Agent(eval_envs, sample_obs=eval_obs).to(device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    else:
        print("Warning: Evaluating untrained agent")
    # Get the average of the weights in the RGB feature extractor (excluding biases)
    rgb_extractor = agent.feature_net.extractors["rgb"]
    weight_sum = 0
    weight_count = 0

    # Iterate through all modules in the RGB extractor
    total_magnitude = 0.0
    total_weights = 0
    for name, param in agent.feature_net.named_parameters():
        if "bias" not in name:
            total_magnitude += torch.sum(torch.abs(param)).item()
            total_weights += param.numel()

    avg_magnitude = total_magnitude / total_weights
    print("CNN Weight magnitude", avg_magnitude)

    # recording evaluation
    if args.capture_video:

        if args.eval_name is not None:
            eval_output_dir = os.getcwd() + "/" + args.eval_name
        else:
            eval_output_dir = "eval"
        print("saving video to", eval_output_dir)
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=False,
            video_fps=args.control_freq,
        )
    for i in range(args.n):
        if i > 0:
            print("Press Enter for eval", i)
            input()
            eval_obs, _ = eval_envs.reset()
            sim_obs, _ = sim_env_test.reset()
        sim_env_test.cube.set_pose(
            sapien.Pose(p=[0.2, 0.02, 0.0], q=[0.0, 0.0, 0.0, 1.0])
        )
        print("diff", sim_env_test.agent.robot.qpos - eval_envs.agent.qpos)
        for step in range(args.num_eval_steps + 1):
            # Small pause to render the image without blocking
            with torch.no_grad():
                agent_obs = eval_obs
                if args.use_sim_obs:
                    agent_obs = sim_obs
                action = agent.get_action(agent_obs, deterministic=True)
                if args.debug:
                    print("Eval step", step + 1)
                    critic_value = agent.get_value(agent_obs)

                    import matplotlib.pyplot as plt

                    # sim_env_test.agent.robot.set_qpos(eval_envs.agent.qpos)
                    # plt.title("Agent RGB Observation")
                    # Create figure and axes only once before the loop
                    if not hasattr(plt, "comparison_fig"):
                        plt.comparison_fig, (plt.ax1, plt.ax2, plt.ax3) = plt.subplots(
                            1, 3, figsize=(10, 6)
                        )

                    # Clear previous images
                    plt.ax1.clear()
                    plt.ax2.clear()
                    plt.ax3.clear()
                    # Update with new images
                    real_img = eval_obs["rgb"][0].cpu().numpy()
                    plt.ax1.imshow(real_img)
                    plt.ax1.set_title("Real Robot Observation")
                    # sim_obs, _ , _, _, _ = sim_env_test.step(None)
                    # sim_obs = dict(rgb=sim_env_test.render_sensors()[:, :128, :128, :])
                    sim_img = sim_obs["rgb"][0].cpu().numpy()
                    plt.ax2.imshow(sim_img)
                    plt.ax2.set_title("Simulation Observation")
                    # Overlay the two images in the third subplot
                    plt.ax3.imshow(real_img, alpha=0.5)
                    plt.ax3.imshow(sim_img, alpha=0.5)
                    plt.ax3.set_title("Overlaid Images")
                    plt.comparison_fig.tight_layout()
                    plt.draw()
                    plt.pause(0.0001)
                    obs = eval_envs.get_obs()
                    print("Agent Obs:")
                    agent_obs = obs["agent"]
                    for value in agent_obs:
                        print(value, agent_obs[value])
                    print("Extra Obs:")
                    extra_obs = obs["extra"]
                    for value in extra_obs:
                        print(value, extra_obs[value])

                    print("action", action)
                    print("value", critic_value)
                    input("Press Enter for next action")
                    print()

                (
                    sim_obs,
                    sim_rew,
                    sim_terminations,
                    sim_truncations,
                    sim_infos,
                ) = sim_env_test.step(action)
                print(sim_env_test.agent.robot.qpos - eval_envs.agent.qpos)
                # sim_env_test.agent.robot.set_qpos(eval_envs.agent.qpos)
                # sim_obs = sim_env_test.get_obs()

                (
                    eval_obs,
                    eval_rew,
                    eval_terminations,
                    eval_truncations,
                    eval_infos,
                ) = eval_envs.step(action)

        # save the video without resetting the env
        if args.capture_video:
            eval_envs.flush_video(verbose=True)

"""
python3 ppo_rgb.py --env_id="GrabCube-v1" --num_envs=256 \
    --update_epochs=4 --num_minibatches=8 --total_timesteps=15_000_000 \
    --num-steps=75 --num_eval_steps=75 --gamma=0.90 \
    --no_partial_reset --render_mode=rgb_array --reconfiguration_freq=1 \
    --obs_mode=rgb+segmentation --seed=0 --no_finite_horizon_gae \
    --exp_name=scratch/grab_cube_training_3 --user_kwargs_path=../../tutorials/user_kwargs.json

"""
