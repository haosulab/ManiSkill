# py-spy record -f speedscope -r 1000 -o profile -- python manualtest/benchmark_gpu_sim.py
# python manualtest/benchmark_orbit_sim.py --task "Isaac-Lift-Cube-Franka-v0" --num_envs 512 --headless
import argparse
import time

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx
import sapien.render
import torch
import tqdm

import mani_skill2.envs
from mani_skill2.envs.scenes.tasks.planner.planner import PickSubtask
from mani_skill2.envs.scenes.tasks.sequential_task import SequentialTaskEnv
from mani_skill2.utils.scene_builder.ai2thor.variants import ArchitecTHORSceneBuilder
from mani_skill2.vector.wrappers.gymnasium import ManiSkillVectorEnv
from profiling import Profiler
from mani_skill2.utils.visualization.misc import images_to_video, tile_images
from mani_skill2.utils.wrappers.flatten import FlattenActionSpaceWrapper


def main(args):
    profiler = Profiler(output_format=args.format)
    num_envs = args.num_envs
    # env = gym.make(
    #     args.env_id,
    #     num_envs=num_envs,
    #     obs_mode=args.obs_mode,
    #     # enable_shadow=True,
    #     render_mode=args.render_mode,
    #     control_mode=args.control_mode,
    #     sim_cfg=dict(control_freq=50)
    # )
    SCENE_IDX_TO_APPLE_PLAN = {
        0: [PickSubtask(obj_id="objects/Apple_5_111")],
        1: [PickSubtask(obj_id="objects/Apple_16_40")],
        2: [PickSubtask(obj_id="objects/Apple_12_64")],
        3: [PickSubtask(obj_id="objects/Apple_29_113")],
        4: [PickSubtask(obj_id="objects/Apple_28_35")],
        5: [PickSubtask(obj_id="objects/Apple_17_88")],
        6: [PickSubtask(obj_id="objects/Apple_1_35")],
        7: [PickSubtask(obj_id="objects/Apple_25_48")],
        8: [PickSubtask(obj_id="objects/Apple_9_46")],
        9: [PickSubtask(obj_id="objects/Apple_13_72")],
    }

    SCENE_IDX = 6
    env: SequentialTaskEnv = gym.make(
        "SequentialTask-v0",
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        control_mode="pd_joint_delta_pos",
        robot_uids="fetch",
        scene_builder_cls=ArchitecTHORSceneBuilder,
        task_plans=[SCENE_IDX_TO_APPLE_PLAN[SCENE_IDX]],
        scene_idxs=SCENE_IDX,
        num_envs=args.num_envs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env)
    sensor_settings_str = []
    for uid, cam in env.base_env._sensors.items():
        cfg = cam.cfg
        sensor_settings_str.append(f"{cfg.width}x{cfg.height}")
    sensor_settings_str = "_".join(sensor_settings_str)
    print(
        "# -------------------------------------------------------------------------- #"
    )
    print(
        f"Benchmarking ManiSkill GPU Simulation with {num_envs} parallel environments"
    )
    print(
        f"env_id={args.env_id}, obs_mode={args.obs_mode}, control_mode={args.control_mode}"
    )
    print(
        f"render_mode={args.render_mode}, sensor_details={sensor_settings_str}, save_video={args.save_video}"
    )
    print(
        f"sim_freq={env.base_env.sim_freq}, control_freq={env.base_env.control_freq}"
    )
    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.base_env.single_action_space}")
    print(
        "# -------------------------------------------------------------------------- #"
    )
    images = []
    video_nrows = int(np.sqrt(num_envs))
    with torch.inference_mode():
        env.reset(seed=2022)
        env.step(env.action_space.sample())  # warmup step
        env.reset(seed=2022)
        if args.save_video:
            images.append(env.render().cpu().numpy())
        N = 100
        with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=env.base_env.device)
                    - 1
                )
                obs, rew, terminated, truncated, info = env.step(actions)
                if args.save_video:
                    images.append(env.render().cpu().numpy())
        profiler.log_stats("env.step")

        if args.save_video:
            images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
            images_to_video(
                images,
                output_dir="./videos/benchmark",
                video_name=f"mani_skill_gpu_sim-{args.env_id}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
                fps=30,
            )
            del images
        env.reset(seed=2022)
        # N = 1000
        # with profiler.profile("env.step+env.reset", total_steps=N, num_envs=num_envs):
        #     for i in range(N):
        #         actions = (
        #             2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        #         )
        #         obs, rew, terminated, truncated, info = env.step(actions)
        #         if i % 200 == 0 and i != 0:
        #             env.reset()
        # profiler.log_stats("env.step+env.reset")
    env.close()
    # append results to csv
    try:
        assert (
            args.save_video == False
        ), "Saving video slows down speed a lot and it will distort results"

        profiler.update_csv(
            "videos/benchmark_results_ms3.csv",
            dict(
                env_id=args.env_id,
                obs_mode=args.obs_mode,
                num_envs=args.num_envs,
                control_mode=args.control_mode,
                sensor_settings=sensor_settings_str,
                gpu_type=torch.cuda.get_device_name()
            ),
        )
    except:
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="state")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("-n", "--num-envs", type=int, default=1024)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="sensors",
        help="which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running.",
    ),
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save videos"
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="stdout",
        help="format of results. Can be stdout or json.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
