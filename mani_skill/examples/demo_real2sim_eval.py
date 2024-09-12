"""
scripts:

## installation
mamba create -n "ms3-octo" "python==3.10.12"
mamba activate ms3-octo
pip install -e . # install mani_skill
pip install torch==2.3.1 tyro==0.8.5 # latest tyro is not compatible with python 3.10.12

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

#!/bin/bash

models=("octo-small" "octo-base", "rt-1x")
env_ids=(
    "PutCarrotOnPlateInScene-v1"
    "PutSpoonOnTableClothInScene-v1"
    "StackGreenCubeOnYellowCubeInScene-v1"
)

for model in "${models[@]}"; do
    for env_id in "${env_ids[@]}"; do
        echo "Running evaluation for model: $model, environment: $env_id"
        XLA_PYTHON_CLIENT_PREALLOCATE=false python mani_skill/examples/demo_real2sim_eval.py \
            --model="$model" -e "$env_id" -s 0 --num-episodes 72
    done
done

echo "All evaluations completed."


XLA_PYTHON_CLIENT_PREALLOCATE=false python mani_skill/examples/demo_octo_eval.py \
    --model="octo-small" -e "StackGreenCubeOnYellowCubeInScene-v1" -s 0 --num-episodes 24
"""


from collections import defaultdict
import json
import os
import signal
import numpy as np
from typing import Annotated, Optional

import torch
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
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeInScene-v1, PutEggplantInBasketInScene-v1"""

    shader: str = "default"

    num_envs: int = 1
    """Number of environments to run. Currently only 1 is supported at the moment"""

    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

    record_dir: str = "videos"
    """The directory to save videos and results"""

    model: Optional[str] = None
    """The model to evaluate on the given environment. Can be one of octo-base, octo-small, rt-1x. If not given, random actions are sampled."""

    ckpt_path: str = ""
    """Checkpoint path for models. Only used for RT models"""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    reset_by_episode_id: bool = True
    """Whether to reset by fixed episode ids instead of random sampling initial states."""

    info_on_video: bool = False
    """Whether to write info text onto the video"""

    save_video: bool = True
    """Whether to save videos"""

    debug: bool = False
def parse_observation(obs):
    img =  common.to_numpy(obs["sensor_data"]["3rd_view_camera"]["rgb"], dtype=np.uint8)
    return img
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
        sim_backend="gpu",
    )
    # TODO (stao): support GPU evals
    # env = CPUGymWrapper(env)
    # import ipdb; ipdb.set_trace()

    if args.debug:
        renderer = visualization.ImageRenderer(wait_for_button_press=True)
        obs, _ = env.reset(seed=args.seed, options={"episode_id": torch.tensor([args.seed + i for i in range(args.num_envs)])})
        env.render_human().paused=True
        img = parse_observation(obs)
        import ipdb; ipdb.set_trace()
        if len(img) > 1:
            # tile images
            img = np.concatenate(img, axis=1)
        renderer(img)
        while True:
            env.render_human()
            env.step(None)

    model = None
    try:
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        from simpler_env.policies.octo.octo_model import OctoInference
        policy_setup = "widowx_bridge"
        if args.model == "octo-base" or args.model == "octo-small":
            model = OctoInference(model_type=args.model, policy_setup=policy_setup, init_rng=args.seed, action_scale=1)
        elif args.model == "rt-1x":
            ckpt_path=args.ckpt_path
            model = RT1Inference(
                saved_model_path=ckpt_path,
                policy_setup=policy_setup,
                action_scale=1,
            )
        elif args.model is not None:
            raise ValueError(f"Model {args.model} does not exist / is not supported.")
    except:
        if args.model is not None:
            raise Exception("SIMPLER Env Policy Inference is not installed")

    model_name = args.model if args.model is not None else "random"
    if model_name == "random":
        print("Using random actions.")
    exp_dir = os.path.join(args.record_dir, f"real2sim_eval/{model_name}_{args.env_id}")

    eval_metrics = defaultdict(list)
    eps_count = 0
    while eps_count < args.num_episodes:
        seed = args.seed + eps_count
        obs, _ = env.reset(seed=seed, options={"episode_id": torch.tensor([seed + i for i in range(args.num_envs)])})
        instruction = env.unwrapped.get_language_instruction()
        print("instruction:", instruction)
        if model is not None:
            model.reset(instruction)
        images = []
        predicted_terminated, truncated = False, False
        images.append(parse_observation(obs))
        actions=[]
        gt_actions = np.load(os.path.join(exp_dir, f"eval_{seed}_actions.npy"))
        iit = 0
        while not (predicted_terminated or truncated):
            if model is not None:
                action = gt_actions[iit]
                iit += 1
                # raw_action, action = model.step(images[-1][0], instruction)
                # raw_action, action2 = model.step(images[-1][1], instruction)
                # import ipdb; ipdb.set_trace()
                # TODO read the JAX arrays and DL pack them directly
                # predicted_terminated = bool(action["terminate_episode"][0] > 0)
                # action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                # actions.append(action)
                # action2 = np.concatenate([action2["world_vector"], action2["rot_axangle"], action2["gripper"]])
                # action = common.to_tensor(np.stack([action, action2]))
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            truncated = bool(truncated.any())
            if args.info_on_video:
                for i in range(len(images[-1])):
                    images[-1][i] = visualization.put_info_on_image(images[-1][i], info)
            images.append(parse_observation(obs))
        for k, v in info.items():
            eval_metrics[k].append(v.cpu().numpy().flatten())
        if args.save_video:
            for i in range(len(images[-1])):
                images_to_video([img[i] for img in images], exp_dir, f"{'gpu' if env.device.type == 'cuda' else 'cpu'}_eval_{seed + i}_success={info['success'][i].item()}", fps=10, verbose=True)
            # np.save(os.path.join(exp_dir, f"eval_{seed}_actions.npy"), np.stack(actions))
        eps_count += args.num_envs
        if args.num_envs == 1:
            print(f"Evaluated episode {eps_count}. Seed {seed}. Results after {eps_count} episodes:")
        else:
            print(f"Evaluated {args.num_envs} episodes, seeds {seed} to {eps_count}. Results after {eps_count} episodes:")
        for k, v in eval_metrics.items():
            print(f"{k}: {np.mean(v)}")

    mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
    with open(os.path.join(exp_dir, "eval_metrics.json"), "w") as f:
        json.dump(mean_metrics, f)
    print(f"Evaluation complete. Results saved to {exp_dir}")

if __name__ == "__main__":
    main()
