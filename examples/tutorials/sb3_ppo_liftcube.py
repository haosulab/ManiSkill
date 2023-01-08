# Import required packages
import gym
import gym.spaces as spaces
from tqdm.notebook import tqdm
import numpy as np
import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
import h5py
from mani_skill2.utils.io_utils import load_json
import matplotlib.pyplot as plt
import torch.nn as nn
import torch as th
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

"""
SB3 won't be able to use ManiSkill observations out of the box so we can define a custom observation wrapper to 
make the ManiSkill environment conform with SB3. Here, we are simply going to take the two RGB images, 
two depth images from both cameras (base camera and hand camera) and the state data and create a workable observation for SB3.
"""
class ManiSkillRGBDWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_size=(128, 128)) -> None:
        super().__init__(env)
        obs_space = env.observation_space
        self.image_size = image_size
        
        # We want the following states to be kept in the observations. 
        # obs_space is the original environment's observation space
        state_spaces = [
            obs_space["agent"]['base_pose'], # pose of the robot
            obs_space["agent"]['qpos'], # robot configuration position
            obs_space["agent"]['qvel'], # robot configuration velocity
        ]
        for k in obs_space["extra"]:
            # includes gripper pose and goal information depending on environment
            state_spaces.append(obs_space["extra"][k])
        # Define the new state space
        state_size = sum([space.shape[0] for space in state_spaces])
        state_space = spaces.Box(-float("inf"), float("inf"), shape=(state_size, ))
        
        # Get the image dimensions. Note that there is a base_camera and a hand_camera, both of which output the same shape
        h, w, _ = obs_space["image"]["base_camera"]["rgb"].shape
        if self.image_size is not None:
            h, w = self.image_size
        new_shape = (h, w, 8) # the shape is HxWx8, where 8 comes from combining two RGB images and two depth images
        low = np.full(new_shape, -float("inf"))
        high = np.full(new_shape, float("inf"))
        rgbd_space = spaces.Box(low, high, dtype=obs_space["image"]["base_camera"]["rgb"].dtype)
        
        # create the observation space
        self.observation_space = spaces.Dict({
            "rgbd": rgbd_space,
            "state": state_space
        })

    @staticmethod # make this static so both RL and IL tutorials can use this
    def convert_observation(observation, image_size):
        # This function replaces the original observations. We scale down images by 255 and 
        # flatten the states in the original observations
        image_obs = observation["image"]
        rgb = image_obs["base_camera"]["rgb"] / 255.0
        depth = image_obs["base_camera"]["depth"]
        rgb2 = image_obs["hand_camera"]["rgb"] / 255.0
        depth2 = image_obs["hand_camera"]["depth"]
        if image_size is not None and image_size != (rgb.shape[0], rgb.shape[1]):
            rgb = cv2.resize(rgb, image_size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, image_size, interpolation=cv2.INTER_LINEAR)[:,:,None]
            rgb2 = cv2.resize(rgb2, image_size, interpolation=cv2.INTER_LINEAR)
            depth2 = cv2.resize(depth2, image_size, interpolation=cv2.INTER_LINEAR)[:,:,None]
        from mani_skill2.utils.common import flatten_state_dict
        state = np.hstack([
            flatten_state_dict(observation["agent"]),
            flatten_state_dict(observation["extra"])
        ])
        
        # combine the RGB and depth images
        rgbd = np.concatenate([rgb, depth, rgb2, depth2], axis=2)
        obs = dict(rgbd=rgbd, state=state)
        return obs
    def observation(self, observation):
        return ManiSkillRGBDWrapper.convert_observation(observation, image_size=self.image_size)

    def step(self, action):
        o, r, d, info = super().step(action)
        d = False
        if info["elapsed_steps"] >= 100:
            # trunate episodes after 100 steps rather than the default 200 steps for faster training
            # set TimeLimit.truncated to True to tell SB3 it's a truncation and not task success
            info["TimeLimit.truncated"] = True
            d = True
        return o,r,d,info

"""
SB3 natively doesn't support processing RGB data with depth information, so we will need to create a custom network 
to process that data. We can make use of the SB3 BaseExtractor class to do this so we can fit our model into any of SB3's algorithms.
For more details on feature extractors see the SB3 docs: 
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#custom-feature-extractor
"""

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
                    nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten()
                )
                
                # to easily figure out the dimensions after flattening, we pass a test tensor
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
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

if __name__ == "__main__":
    env_id = "LiftCube-v0"
    obs_mode = "rgbd"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "dense"
    
    num_cpu = 8

    def make_env(env_id: str, rank: int, seed: int = 0, record_dir: str = None):
        def _init() -> gym.Env:
            import mani_skill2.envs
            env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode)
            env = ManiSkillRGBDWrapper(env)
            if record_dir is not None:
                env = RecordEpisode(env, record_dir, info_on_video=True, render_mode="cameras")
            env.seed(seed+rank)
            return env
        set_random_seed(seed)
        return _init

    # create one eval environment
    eval_env = SubprocVecEnv([make_env(env_id, i, record_dir="logs/videos") for i in range(1)])
    eval_env = VecMonitor(eval_env) # attach this so SB3 can log reward metrics
    eval_env.reset()

    # create num_cpu training environments
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = VecMonitor(env)
    obs = env.reset()

    # define callbacks to periodically save our model and evaluate it to help monitor training
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                         log_path="./logs/", eval_freq=4000,
                         deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(
        save_freq=4000,
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # Define the policy configuration and algorithm configuration
    policy_kwargs = dict(features_extractor_class=CustomExtractor, net_arch=[128, 128])
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
        n_steps=400, batch_size=400,
        gamma=0.8,
        n_epochs=5,
        tensorboard_log="./logs",
        target_kl=0.2, 
        ent_coef=0,
        max_grad_norm=0.5,
        learning_rate=3e-4
    )
    # Train an agent with PPO for 1,000,000 interactions
    model.learn(300_000, callback=[checkpoint_callback, eval_callback],)
    # Save the final model
    model.save("./logs/latest_model")

    # Load the saved model
    model = model.load("./logs/latest_model")

    # Evaluate the model
    results = evaluate_policy(model, eval_env, deterministic=True, render=False, return_episode_rewards=True, n_eval_episodes=10)
    print(results)