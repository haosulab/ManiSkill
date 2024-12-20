from pathlib import Path
from typing import Annotated
import genesis as gs
import torch
import numpy as np
import gymnasium as gym
import envs.genesis
########################## init ##########################
gs.init(backend=gs.gpu, logging_level="warning")

from dataclasses import dataclass
import tyro
from profiling import Profiler, images_to_video

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "FrankaMoveBenchmark-v1"
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "state"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_target_delta_pos"
    """The control mode to use. There is pd_joint_target_delta_pos, pd_joint_delta_pos, and pd_joint_pos.

    # TODO (stao): add pd_joint_target_delta_pos to genesis benchmark

    pd_joint_target_delta_pos is most similar to ManiSkill's pd_joint_delta_pos (the typical default)
    in terms of behavior since ManiSkill disables robot gravity for simplicity whereas genesis currently does not
    support disabling gravity on specific articulation/robot links. Without gravity disabled using a pd_joint_delta_pos controller
    will typically cause the robot to slowly fall downwards since PD controller undershoots the target position.
    """
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1024
    cpu_sim: bool = False
    seed: int = 0
    save_example_image: bool = False
    control_freq: int | None = 60
    sim_freq: int | None = 120
    num_cams: int | None = None
    cam_width: int | None = None
    cam_height: int | None = None
    render_mode: str = "rgb_array"
    save_video: bool = False
    save_results: str | None = None

def main(args: Args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    env = gym.make(args.env_id, num_envs=num_envs, sim_freq=args.sim_freq, control_mode=args.control_mode, control_freq=args.control_freq, render_mode=args.render_mode)

    obs, _ = env.reset()
    env.step(torch.zeros(env.action_space.shape, device=gs.device)) # take one step in case genesis has some warm-start delays
    obs, _ = env.reset()
    N = 1000
    if args.save_video:
        images = [env.unwrapped.render_rgb_array()]
    with torch.inference_mode():
        with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=gs.device)
                    - 1
                )
                obs, _, _, _, _ = env.step(actions)
                if args.save_video:
                    rgb = env.unwrapped.render_rgb_array()
                    images.append(rgb)
        env.close()
        profiler.log_stats("env.step")
        if args.save_video:
            images_to_video(
                images,
                output_dir="./videos/genesis_benchmark",
                video_name=f"genesis_gpu_sim-random_actions-{args.env_id}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
                fps=30,
            )
            del images
    # if environment has some predefined actions run those
    for k, v in env.unwrapped.fixed_trajectory.items():
        obs, _ = env.reset()
        env.step(torch.zeros(env.action_space.shape, device=gs.device)) # take one step in case genesis has some warm-start delays
        obs, _ = env.reset()
        if args.save_video:
            images = [env.unwrapped.render_rgb_array()]
        actions = v["actions"]
        if v["control_mode"] == "pd_joint_pos":
            env.unwrapped.set_control_mode(v["control_mode"])
            N = v["shake_steps"] if "shake_steps" in v else 0
            N += sum([a[1] for a in actions])
            with profiler.profile(f"{k}_env.step", total_steps=N, num_envs=num_envs):
                i = 0
                for action in actions:
                    for _ in range(action[1]):
                        env.step(torch.tile(action[0], (num_envs, 1)))
                        i += 1
                        if args.save_video:
                            rgb = env.unwrapped.render_rgb_array()
                            images.append(rgb)
                # runs a "shake" test, typically used to check stability of contacts/grasping
                if "shake_steps" in v:
                    env.unwrapped.set_control_mode("pd_joint_target_delta_pos")
                    while i < N:
                        actions = v["shake_action_fn"]()
                        env.step(actions)
                        if args.save_video:
                            rgb = env.unwrapped.render_rgb_array()
                            images.append(rgb)
                        i += 1
            profiler.log_stats(f"{k}_env.step")
            if args.save_video:
                images_to_video(
                    images,
                    output_dir="./videos/genesis_benchmark",
                    video_name=f"genesis_gpu_sim-fixed_trajectory={k}-{args.env_id}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
                    fps=30,
                )
                del images
    env.close()
    if args.save_results:
        # append results to csv
        # try:
        assert (
            args.save_video == False
        ), "Saving video slows down speed a lot and it will distort results"
        Path("benchmark_results").mkdir(parents=True, exist_ok=True)
        data = dict(
            env_id=args.env_id,
            obs_mode=args.obs_mode,
            num_envs=args.num_envs,
            control_mode=args.control_mode,
            gpu_type=torch.cuda.get_device_name()
        )
        data.update(
            num_cameras=args.num_cams,
            camera_width=args.cam_width,
            camera_height=args.cam_height,
        )
        profiler.update_csv(
            args.save_results,
            data,
        )
        # except:
        #     pass
if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
