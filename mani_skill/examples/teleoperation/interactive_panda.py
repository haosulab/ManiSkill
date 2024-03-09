import argparse
from ast import parse
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
def main(args):
    output_dir = f"{args.record_dir}/teleop/{args.env_id}"
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
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system"
    )
    num_trajs = 0
    seed = 0
    env.reset(seed=seed)
    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"saving videos to {output_dir}")

    trajectory_data = h5py.File(h5_file_path)
    with open(json_file_path, "r") as f:
        json_data = json.load(f)
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
        shader_dir="rt-med",
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
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
        env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

        env.base_env.set_state_dict(env_states_list[0])
        for action in np.array(data["actions"]):
            env.step(action)

    trajectory_data.close()
    env.close()
    del env



def solve(env: BaseEnv, debug=False, vis=False):
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
    viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "panda_hand")._objs[0].entity)
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
    while True:

        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            # TODO (stao): print help menu
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost panda arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data and save videos
            """)
            pass
        # elif viewer.window.key_press("k"):
        #     print("Saving checkpoint")
        #     last_checkpoint_state = env.get_state_dict()
        # elif viewer.window.key_press("l"):
        #     if last_checkpoint_state is not None:
        #         print("Loading previous checkpoint")
        #         env.set_state_dict(last_checkpoint_state)
        #     else:
        #         print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        # elif viewer.window.key_press("r"):
        #     viewer.select_entity(None)
        #     return "restart"
        # elif viewer.window.key_press("t"):
        #     # TODO (stao): change from position transform to rotation transform
        #     pass
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g"):
            if gripper_open:
                gripper_open = False
                _, reward, _ ,_, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _ ,_, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
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
            result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.102]), dry_run=True)
            if result != -1 and len(result["position"]) < 100:
                _, reward, _ ,_, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-r", "--robot-uid", type=str, default="panda", help="Robot setups supported are ['panda']")
    parser.add_argument("--record-dir", type=str, default="demos")
    args, opts = parser.parse_known_args()

    return args
if __name__ == "__main__":
    main(parse_args())
