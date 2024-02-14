import argparse
from ast import parse
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from tqdm import tqdm
from mani_skill2.envs.sapien_env import BaseEnv

from mani_skill2.envs.tasks.pick_cube import PickCubeEnv
from mani_skill2.examples.motionplanning.motionplanner import \
    PandaArmMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
from mani_skill2.utils.wrappers.record import RecordEpisode
def main(args):
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
        # shader_dir="rt-fast",
    )
    # TODO (don't record episode directly, its slow. Just save trajectory. Then use the actions/states and then re-run them to generate videos if asked)
    env = RecordEpisode(
        env,
        output_dir=f"videos/teleop-{args.env_id}",
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
    )
    num_trajs = 0
    for seed in range(100):
        num_trajs += 1
        code = solve(env, seed=seed, debug=False, vis=True)
        if code == "quit": break
        elif code == "continue": continue
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print("saving videos")

    trajectory_data = h5py.File(h5_file_path)
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
        # shader_dir="rt-fast",
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/teleop-{args.env_id}",
        trajectory_name="trajectory",
        save_video=True,
        info_on_video=False,
        save_trajectory=False,
        video_fps=30
    )
    for episode in json_data["episodes"]:
        traj_id = f"traj_{episode['episode_id']}"
        data = trajectory_data[traj_id]
        env.reset(**episode["reset_kwargs"])
        env._base_env.set_state(data["env_states"][0])
        for action in np.array(data["actions"]):
            env.step(action)

    trajectory_data.close()
    env.close()
    del env



def solve(env: BaseEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_acc_limits=0.5,
        joint_vel_limits=0.5,
    )
    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True

    while True:
        # viewer.select_entity(planner.grasp_pose_visual._objs[0])
        transform_window = viewer.plugins[0]

        transform_window: sapien.utils.viewer.viewer.TransformWindow
        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("k"):
            print("Saving checkpoint")
            last_checkpoint_state = env.get_state()
        elif viewer.window.key_press("l"):
            if last_checkpoint_state is not None:
                print("Loading previous checkpoint")
                env.set_state(last_checkpoint_state)
            else:
                print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        if viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g"):
            if gripper_open:
                gripper_open = False
                planner.close_gripper()
            else:
                gripper_open = True
                planner.open_gripper()
        # # TODO left, right depend on orientation really.
        # elif viewer.window.key_press("down"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, 0, 0.01]))
        # elif viewer.window.key_press("up"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, 0, -0.01]))
        # elif viewer.window.key_press("right"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, -0.01, 0]))
        # elif viewer.window.key_press("left"):
        #     pose = planner.grasp_pose_visual.pose
        #     planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, +0.01, 0]))
        if execute_current_pose:
            # z-offset of end-effector gizmo to TCP position is hardcoded for the panda robot here
            result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.102]))
            execute_current_pose = False
            print(f"Reward: {result[1]}, Info: {result[-1]}")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    # parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    # parser.add_argument("--render-mode", type=str, default="cameras")
    # parser.add_argument("--enable-sapien-viewer", action="store_true")
    # parser.add_argument("--record-dir", type=str)
    args, opts = parser.parse_known_args()

    # # Parse env kwargs
    # print("opts:", opts)
    # eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    # env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    # print("env_kwargs:", env_kwargs)
    # args.env_kwargs = env_kwargs

    return args
if __name__ == "__main__":
    main(parse_args())
