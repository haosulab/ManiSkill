"""
scripts:

## installation
mamba create -n "ms3-octo" "python==3.10.12"
mamba activate ms3-octo
pip install -e . # install mani_skill
pip install torch==2.3.1

git clone https://github.com/simpler-env/SimplerEnv
cd SimplerEnv
pip install -e .
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support

pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
git clone https://github.com/octo-models/octo/
cd octo
git checkout 653c54acde686fde619855f2eac0dd6edad7116b  # we use octo-1.0
pip install -e .
"""


from collections import defaultdict
import json
import os
import signal
import numpy as np
from typing import Annotated
from mani_skill.utils import common
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1"""

    shader: str = "default"

    num_envs: int = 1
    """Number of environments to run. Currently only 1 is supported at the moment"""

    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

    record_dir: str = "videos"
    """The directory to save videos and results"""

    model: str = "octo-base"
    """The model to evaluate on the given environment. Can be one of octo-base or octo-small"""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the random actions and environment. Default seed is 0"""

    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

def main():
    args = tyro.cli(Args)
    if args.seed is not None:
        np.random.seed(args.seed)


    sensor_configs = dict()
    sensor_configs["shader_pack"] = args.shader
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        num_envs=args.num_envs,
        sensor_configs=sensor_configs,
        sim_backend="cpu",
    )
    # TODO (stao): support GPU evals
    env = CPUGymWrapper(env)

    obs, _ = env.reset(seed=args.seed)
    n_cams = 0
    for config in env.unwrapped._sensors.values():
        if isinstance(config, Camera):
            n_cams += 1
    print(f"Visualizing {n_cams} RGBD cameras")

    from simpler_env.policies.octo.octo_model import OctoInference
    model_name = "octo-small"
    policy_setup = "widowx_bridge"
    model = OctoInference(model_type=model_name, policy_setup=policy_setup, init_rng=0)
    exp_dir = os.path.join(args.record_dir, f"real2sim_eval/{model_name}_{args.env_id}")

    # renderer = visualization.ImageRenderer(wait_for_button_press=False)
    def render_obs(obs):
        img =  common.to_numpy(obs["sensor_data"]["3rd_view_camera"]["rgb"], dtype=np.uint8)
        # renderer(img)
        return img

    eval_metrics = defaultdict(list)
    eps_count = 0
    for seed in range(args.seed, args.seed+args.num_episodes):
        obs, _ = env.reset(seed=seed)
        instruction = env.unwrapped.get_language_instruction()
        print("instruction:", instruction)
        model.reset(instruction)
        images = []
        images.append(render_obs(obs))
        predicted_terminated, truncated = False, False
        while not (predicted_terminated or truncated):
            raw_action, action = model.step(images[-1])
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            obs, reward, terminated, truncated, info = env.step(action)
            truncated = bool(truncated)
            img = render_obs(obs)
            img = visualization.put_info_on_image(img, info)
            images.append(img)
        for k, v in info.items():
            eval_metrics[k].append(v)
        images_to_video(images, exp_dir, f"octo_eval_{seed}", fps=10, verbose=True)
        eps_count += 1
        print(f"Evaluated episode {eps_count}. Seed {seed}. Results after {eps_count} episodes:")
        for k, v in eval_metrics.items():
            print(f"{k}: {np.mean(v)}")

    mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
    with open(os.path.join(exp_dir, "eval_metrics.json"), "w") as f:
        json.dump(mean_metrics, f)

if __name__ == "__main__":
    main()
