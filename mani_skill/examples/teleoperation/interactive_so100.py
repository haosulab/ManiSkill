import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.examples.motionplanning.so100.motionplanner import \
    SO100ArmMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "so100"
    """The robot to use. Robot setups supported for teleop in this script is so100"""
    record_dir: str = "demos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = False
    """whether to save the videos of the demonstrations after collecting them all"""
    viewer_shader: str = "rt-fast"
    """the shader to use for the viewer. 'default' is fast but lower-quality shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    video_saving_shader: str = "rt-fast"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""

def parse_args() -> Args:
    return tyro.cli(Args)

def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        robot_uids=args.robot_uid, 
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader)
    )

    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=True,
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
    print(f"Trajectories saved to {h5_file_path}")
    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="none",
            robot_uids=args.robot_uid, 
            human_render_camera_configs=dict(shader_pack=args.video_saving_shader),
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
    robot_has_gripper = True
    planner = SO100ArmMotionPlanningSolver(
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
    def select_so100_hand():
        viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "Fixed_Jaw")._objs[0].entity) 
    select_so100_hand()
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
    while True:
        transform_window.enabled = True

        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open (if there is a gripper)
            u: move the so100 hand up
            j: move the so100 hand down
            arrow_keys: move the so100 hand in the direction of the arrow keys
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost so100 arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            """)
            pass
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                _, reward, _ ,_, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _ ,_, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        elif viewer.window.key_press("u"):
            select_so100_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_so100_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_so100_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[+0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_so100_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[-0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_so100_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, -0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_so100_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, +0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        if execute_current_pose:
            fixed_jaw_tip_position = transform_window._gizmo_pose * sapien.Pose(p=[0.01, -0.097, 0])
            target_pose = sapien.Pose(p=fixed_jaw_tip_position.p - planner._so_100_grasp_pose_tcp_transform.p, q=fixed_jaw_tip_position.q)

            result = planner.move_to_pose_with_screw(target_pose, dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False


if __name__ == "__main__":
    main(parse_args())