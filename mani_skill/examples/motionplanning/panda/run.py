import argparse
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import os.path as osp
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.examples.motionplanning.panda.solutions import solvePickCube, solveStackCube
MP_SOLUTIONS = {
    "PickCube-v1": solvePickCube,
    "StackCube-v1": solveStackCube,
    # "PegInsertionSide-v1": solvePegInsertionSide
}
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")
    parser.add_argument("-o", "--obs-mode", type=str, default="none", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=10, help="Number of trajectories to generate.")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos/motionplanning", help="where to save the recorded trajectories")
    return parser.parse_args()

def main(args):
    env_id = args.env_id
    env = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="dense" if args.reward_mode is None else args.reward_mode,
        shader_dir=args.shader
    )
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(f"No already written motion planning solutions for {env_id}. Available options are {list(MP_SOLUTIONS.keys())}")
    env = RecordEpisode(env, output_dir=osp.join(args.record_dir, env_id), trajectory_name=args.traj_name, save_video=args.save_video, source_type="motionplanning", source_desc="official motion planning solution from ManiSkill contributors", video_fps=30)
    solve = MP_SOLUTIONS[env_id]
    print(f"Motion Planning Running on {env_id}")

    for seed in tqdm(range(args.num_traj)):
        res = solve(env, seed=seed, debug=False, vis=True if args.vis else False)
        print(res[-1])
    env.close()

if __name__ == "__main__":
    main(parse_args())
