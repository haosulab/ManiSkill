"""
Code to mass generate demonstrations via motion planning solutions for various tasks

Run python mani_skill /examples/motionplanning/record_demos.py to test it out, which will generate one successful video per task with a MP solution

Run python mani_skill /examples/motionplanning/record_demos.py --no-video -n 1000 to mass generate a demo dataset

"""


import argparse
import os.path as osp

import gymnasium as gym
from tqdm import tqdm

from mani_skill.examples.motionplanning.ms2_tasks.lift_cube import \
    solve as lift_cube_solve
from mani_skill.examples.motionplanning.ms2_tasks.panda_avoid_obstacles import \
    solve as panda_avoid_obstacles_solve
from mani_skill.examples.motionplanning.ms2_tasks.peg_insertion_side import \
    solve as peg_insertion_side_solve
from mani_skill.examples.motionplanning.ms2_tasks.pick_clutter import \
    solve as pick_clutter_solve
from mani_skill.examples.motionplanning.ms2_tasks.pick_cube import \
    solve as pick_cube_solve
from mani_skill.examples.motionplanning.ms2_tasks.plug_charger import \
    solve as plug_charger_solve
from mani_skill.examples.motionplanning.ms2_tasks.stack_cube import \
    solve as stack_cube_solve
from mani_skill.utils.wrappers.record import RecordEpisode


def record_ms2_motion_planned_demonstrations(args):
    MS2_TASKS = [
        ("PickCube-v0", pick_cube_solve),
        ("StackCube-v0", stack_cube_solve),
        ("LiftCube-v0", lift_cube_solve),
        ("PandaAvoidObstacles-v0", panda_avoid_obstacles_solve),
        ("PegInsertionSide-v0", peg_insertion_side_solve),
        ("PlugCharger-v0", plug_charger_solve),
    ]
    for task, solve in MS2_TASKS:
        env = gym.make(
            task,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            reward_mode="sparse",
            shader_dir=args.shader_dir,
        )
        env = RecordEpisode(
            env,
            output_dir=osp.join(args.record_dir, "rigid_body", task),
            trajectory_name="trajectory",
            save_video=(not args.no_video),
            save_on_reset=False,
        )
        print(f"Generating demonstrations for {task}")
        pbar = tqdm(total=args.num_episodes)
        n_success = 0
        n = 0
        seed = args.start_seed
        while n_success < args.num_episodes:
            res = solve(env, seed=seed, debug=False, vis=False)
            if res == -1:
                continue
            else:
                final_obs, reward, terminated, truncated, info = res
            success = info["success"]
            if not args.allow_timeout:
                success = success and (not truncated)
            # Save video
            if not args.no_video and (success or args.allow_failed_videos):
                elapsed_steps = info["elapsed_steps"]
                suffix = "seed={}-success={}-steps={}".format(
                    seed, success, elapsed_steps
                )
                env.flush_video(suffix, verbose=True)
            # Save trajectory
            if success or args.allow_failure:
                env.flush_trajectory(verbose=False)
                pbar.update()

            n_success += int(success)
            n += 1
            seed += 1
        print(f"{task} Success Rate:", n_success / n)
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("-r", "--record-dir", type=str, default="videos")
    parser.add_argument("-n", "--num-episodes", type=int, default=1)
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--no-traj", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--allow-failure", action="store_true")
    parser.add_argument("--allow-failed-videos", action="store_true")
    parser.add_argument("--allow-timeout", action="store_true")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--shader-dir", type=str, default="default")
    args = parser.parse_args()
    record_ms2_motion_planned_demonstrations(args)


if __name__ == "__main__":
    main()
