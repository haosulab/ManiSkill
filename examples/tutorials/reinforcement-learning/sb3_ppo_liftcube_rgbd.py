# Import required packages
import argparse
import os.path as osp
from functools import partial

import gym
import gym.spaces as spaces
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import mani_skill2.envs
from mani_skill2.utils.common import flatten_dict_space_keys, flatten_state_dict
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector import VecEnv
from mani_skill2.vector import make as make_vec_env
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from mani_skill2.vector.wrappers.sb3 import SB3VecEnvWrapper


# Defines a continuous, infinite horizon, task where done is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        return super().reset()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, done, info


class ManiSkillRGBDWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
        self.observation_space = self.init_observation_space(env.observation_space)

    @staticmethod
    def init_observation_space(obs_space: spaces.Dict):
        # States include robot proprioception (agent) and task information (extra)
        # NOTE: SB3 does not support nested observation spaces, so we convert them to flat spaces
        state_spaces = []
        state_spaces.extend(flatten_dict_space_keys(obs_space["agent"]).spaces.values())
        state_spaces.extend(flatten_dict_space_keys(obs_space["extra"]).spaces.values())
        # Concatenate all the state spaces
        state_size = sum([space.shape[0] for space in state_spaces])
        state_space = spaces.Box(-np.inf, np.inf, shape=(state_size,))

        # Concatenate all the image spaces
        image_shapes = []
        for cam_uid in obs_space["image"]:
            cam_space = obs_space["image"][cam_uid]
            image_shapes.append(cam_space["rgb"].shape)
            image_shapes.append(cam_space["depth"].shape)
        image_shapes = np.array(image_shapes)
        assert np.all(image_shapes[0, :2] == image_shapes[:, :2]), image_shapes
        h, w = image_shapes[0, :2]
        c = image_shapes[:, 2].sum(0)
        rgbd_space = spaces.Box(0, np.inf, shape=(h, w, c))

        # Create the new observation space
        return spaces.Dict({"rgbd": rgbd_space, "state": state_space})

    @staticmethod
    def convert_observation(observation):
        # Process images. RGB is normalized to [0, 1].
        images = []
        for cam_uid, cam_obs in observation["image"].items():
            rgb = cam_obs["rgb"] / 255.0
            depth = cam_obs["depth"]

            # NOTE: SB3 does not support GPU tensors, so we transfer them to CPU.
            # For other RL frameworks that natively support GPU tensors, this step is not necessary.
            if isinstance(rgb, th.Tensor):
                rgb = rgb.to(device="cpu", non_blocking=True)
            if isinstance(depth, th.Tensor):
                depth = depth.to(device="cpu", non_blocking=True)

            images.append(rgb)
            images.append(depth)

        # Concatenate all the images
        rgbd = np.concatenate(images, axis=-1)

        # Concatenate all the states
        state = np.hstack(
            [
                flatten_state_dict(observation["agent"]),
                flatten_state_dict(observation["extra"]),
            ]
        )

        return dict(rgbd=rgbd, state=state)

    def observation(self, observation):
        return self.convert_observation(observation)


# We separately define an VecEnv observation wrapper for the ManiSkill VecEnv
# as the gpu optimization makes it incompatible with the SB3 wrapper
class ManiSkillRGBDVecEnvWrapper(VecEnvObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.obs_mode == "rgbd"
        self.observation_space = ManiSkillRGBDWrapper.init_observation_space(
            env.observation_space
        )

    def observation(self, observation):
        return ManiSkillRGBDWrapper.convert_observation(observation)


class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128

        for key, subspace in observation_space.spaces.items():
            # We go through all subspaces in the observation space.
            # We know there will only be "rgbd" and "state", so we handle those below
            if key == "rgbd":
                # here we use a NatureCNN architecture to process images, but any architecture is permissble here
                in_channels = subspace.shape[-1]
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
                        in_channels=32,
                        out_channels=64,
                        kernel_size=4,
                        stride=2,
                        padding=0,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                    ),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                # to easily figure out the dimensions after flattening, we pass a test tensor
                test_tensor = th.zeros(
                    [subspace.shape[2], subspace.shape[0], subspace.shape[1]]
                )
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors["rgbd"] = nn.Sequential(cnn, fc)
                total_concat_size += feature_size
            elif key == "state":
                # for state data we simply pass it through a single linear layer
                state_size = subspace.shape[0]
                extractors["state"] = nn.Linear(state_size, 64)
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "rgbd":
                observations[key] = observations[key].permute((0, 3, 1, 2))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=100,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=160_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    env_id = args.env_id
    num_envs = args.n_envs
    log_dir = args.log_dir
    max_episode_steps = args.max_episode_steps
    total_timesteps = args.total_timesteps
    rollout_steps = 3200

    obs_mode = "rgbd"
    # NOTE: The end-effector space controller is usually more friendly to pick-and-place tasks
    control_mode = "pd_ee_delta_pose"
    use_ms2_vec_env = True

    if args.seed is not None:
        set_random_seed(args.seed)

    # define a make_env function for Stable Baselines
    def make_env(
        env_id: str,
        max_episode_steps=None,
        record_dir: str = None,
    ):
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill2.envs

        env = gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode)
        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env, max_episode_steps)
        env = ManiSkillRGBDWrapper(env)
        # For evaluation, we record videos
        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            env = RecordEpisode(
                env,
                record_dir,
                save_trajectory=False,
                info_on_video=True,
                render_mode="cameras",
            )
        return env

    # Create an environment for evaluation
    if args.eval:
        record_dir = osp.join(log_dir, "videos/eval")
    else:
        record_dir = osp.join(log_dir, "videos")
    env_fn = partial(
        make_env,
        env_id,
        record_dir=record_dir,
    )
    eval_env = SubprocVecEnv([env_fn for _ in range(1)])
    eval_env = VecMonitor(eval_env)  # Attach a monitor to log episode info
    eval_env.seed(args.seed)

    if args.eval:
        env = eval_env
    else:
        # Create vectorized environments for training
        if use_ms2_vec_env:
            env: VecEnv = make_vec_env(
                env_id,
                num_envs,
                obs_mode=obs_mode,
                control_mode=control_mode,
                # specify wrappers for each individual environment e.g here we specify the
                # Continuous task wrapper and pass in the max_episode_steps parameter via the partial tool
                wrappers=[
                    partial(ContinuousTaskWrapper, max_episode_steps=max_episode_steps)
                ],
            )
            env = ManiSkillRGBDVecEnvWrapper(env)
            env = SB3VecEnvWrapper(env)
        else:
            env_fn = partial(
                make_env,
                env_id,
                max_episode_steps=max_episode_steps,
            )
            env = SubprocVecEnv([env_fn for _ in range(num_envs)])
        # Attach a monitor to log episode info
        env = VecMonitor(env)
        env.seed(args.seed)

    # Define the policy configuration and algorithm configuration
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor, net_arch=[256, 128], log_std_init=-0.5
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=rollout_steps // num_envs,
        batch_size=400,
        n_epochs=5,
        gamma=0.8,
        target_kl=0.2,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    if args.eval:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "latest_model")
        # Load the saved model
        model = model.load(model_path)
    else:
        # Define callbacks to periodically save our model and evaluate it to help monitor training
        checkpoint_callback = CheckpointCallback(
            save_freq=10 * rollout_steps // num_envs,
            save_path=log_dir,
        )
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=10 * rollout_steps // num_envs,
            log_path=log_dir,
            best_model_save_path=log_dir,
            deterministic=True,
            render=False,
        )

        # Train an agent with PPO
        model.learn(total_timesteps, callback=[checkpoint_callback, eval_callback])
        # Save the final model
        model.save(osp.join(log_dir, "latest_model"))

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=10,
    )
    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print("Success Rate:", success_rate)


if __name__ == "__main__":
    main()
