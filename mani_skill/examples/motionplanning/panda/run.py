import multiprocessing as mp
import os
from copy import deepcopy
import time
import traceback
import argparse
import gymnasium as gym
import json
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.solutions import solvePushCube, solvePickCube, solveStackCube, solvePegInsertionSide, solvePlugCharger, solvePullCubeTool, solveLiftPegUpright, solvePullCube, solveDrawTriangle, solveDrawSVG, solvePlaceSphere, solveOpenDrawer
from mani_skill.envs.distraction_set import DISTRACTION_SETS

MP_SOLUTIONS = {
    "DrawTriangle-v1": solveDrawTriangle,
    "PickCube-v1": solvePickCube,
    "PickCubeMP-v1": solvePickCube,
    "StackCube-v1": solveStackCube,
    "PegInsertionSide-v1": solvePegInsertionSide,
    "PlugCharger-v1": solvePlugCharger,
    "PushCube-v1": solvePushCube,
    "PullCubeTool-v1": solvePullCubeTool,
    "LiftPegUpright-v1": solveLiftPegUpright,
    "PullCube-v1": solvePullCube,
    "DrawSVG-v1" : solveDrawSVG,
    # 

    # New tasks:
    "LiftPegUpright-v2": solveLiftPegUpright,
    "OpenDrawer-v1": solveOpenDrawer,               # new
    "PlaceSphere-v2": solvePlaceSphere,             # new  
    # "PickCube-v2": solvePickCube,                   # new
    # "PickCube-v3": solvePickCube,                   # new
    # "PickCube-v4": solvePickCube,                   # new
    # "PickCube-v3-VisibleSphere": solvePickCube,     # new
    "StackCube-v2": solveStackCube,                 # new
    # "PlugCharger-v2": solvePlugCharger,             # new
    "PushCube-v2": solvePushCube,                   # new
    "PullCube-v2": solvePullCube,                   # new
    # "PullCubeTool-v2": solvePullCubeTool,           # new
    "PegInsertionSide-v2": solvePegInsertionSide,   # new
}

"""
ENV_ID=PushCube-v2

python mani_skill/examples/motionplanning/panda/run.py \
    --camera-width 640 --camera-height 480 \
    --env-id ${ENV_ID} \
    --num-traj 10 \
    --distraction-set "none" \
    --num-procs 1 \
    --reward-mode "sparse" \
    --random-seed \
    --vis \
    --save-video

    python -m mani_skill.examples.demo_random_action -e ${ENV_ID} --render-mode="human" --shader="rt-fast" --seed 3 --reward_mode "sparse" --pause
"""


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--random-seed", action="store_true", help="whether or not to randomize the seed for each process")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--camera-width", type=int, required=False, help="Width of the camera.")
    parser.add_argument("--camera-height", type=int,  required=False, help="Height of the camera.")
    parser.add_argument("--distraction-set", type=str, required=True, help=f"Distraction set to use. Available options are {list(DISTRACTION_SETS.keys())}")
    return parser.parse_args()

def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    if args.camera_width is not None and args.camera_height is not None:
        distraction_set = DISTRACTION_SETS[args.distraction_set.upper()]
        print("Distraction set:")
        print(json.dumps(distraction_set.to_dict(), indent=2))
        env = gym.make(
            env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode=args.render_mode,
            reward_mode="dense" if args.reward_mode is None else args.reward_mode,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            sim_backend=args.sim_backend,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            distraction_set=distraction_set
        )
    else:
        env = gym.make(
            env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode=args.render_mode,
            reward_mode="dense" if args.reward_mode is None else args.reward_mode,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            sim_backend=args.sim_backend
        )
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")

    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name

    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name, save_video=args.save_video,
        source_type="motionplanning",
        source_desc="official motion planning solution from ManiSkill contributors",
        video_fps=30,
        record_reward=False,
        save_on_reset=False
    )
    output_h5_path = env._h5_file.filename
    solve = MP_SOLUTIONS[env_id]
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    print(f"Motion Planning Running on {env_id} with seed {seed}")
    successes = []
    solution_episode_lengths = []
    failed_motion_plans = 0
    passed = 0
    while True:
        env.reset(seed=seed, options={"reconfigure": True}) # reconfigure so distractor variations are resampled
        res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
        # try:
        # except Exception as e:
        #     print(f"Cannot find valid solution because of an error in motion planning solution: {e}")
        #     print("Traceback:")
        #     print(''.join(traceback.format_tb(e.__traceback__)))
        #     res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)
        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue
        else:
            env.flush_trajectory()
            if args.save_video:
                env.flush_video()
            pbar.update(1)
            pbar.set_postfix(
                dict(
                    success_rate=np.mean(successes),
                    failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                    avg_episode_length=np.mean(solution_episode_lengths),
                    max_episode_length=np.max(solution_episode_lengths),
                    min_episode_length=np.min(solution_episode_lengths)
                )
            )
            seed += 1
            passed += 1
            if passed == args.num_traj:
                break
    env.close()
    return output_h5_path

def main(args):
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("Number of trajectories should be greater than or equal to number of processes")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        # Merge trajectory files
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        if args.random_seed:
            seed = np.random.randint(0, int(2**32-1))
        else:
            seed = 0
        _main(args, start_seed=seed)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
