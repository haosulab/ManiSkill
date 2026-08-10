"""
Generic cross-platform teleoperation script.

Works with any robot that defines `ee_link_name` and `arm_joint_names`.
Uses Pinocchio-based IK (no mplib dependency), so runs on macOS + Linux.

Usage:
    python -m mani_skill.examples.teleoperation.teleop -e PickCube-v1 -r panda
    python -m mani_skill.examples.teleoperation.teleop -e PickCube-v1 -r xarm6_robotiq
    python -m mani_skill.examples.teleoperation.teleop -e PickCube-v1 -r so100
"""

from dataclasses import dataclass, field
from typing import Annotated, Optional

import platform
import gymnasium as gym
import h5py
import json
import numpy as np
import sapien.core as sapien
import sapien.internal_renderer as R
import sapien.utils.viewer
from sapien.utils.viewer.plugin import Plugin
import torch
import tyro

_IS_MACOS = platform.system() == "Darwin"
_DEFAULT_SHADER = "default" if _IS_MACOS else "rt-fast"

from mani_skill import format_path
from mani_skill.agents.controllers.base_controller import CombinedController
from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose as MSPose
from mani_skill.utils.wrappers.record import RecordEpisode
import mani_skill.trajectory.utils as trajectory_utils


class TeleopPlugin(Plugin):
    """Custom viewer plugin that locks selection to the gizmo link and provides a gripper slider."""

    def __init__(self):
        self._locked_entity = None
        self._ui_window = None
        self.gripper_openness = 1.0  # 1.0 = fully open, 0.0 = fully closed
        self._has_gripper = False

    def set_locked_entity(self, entity):
        self._locked_entity = entity

    def set_has_gripper(self, has_gripper):
        self._has_gripper = has_gripper
        self._ui_window = None  # force rebuild

    def notify_selected_entity_change(self):
        if self._locked_entity is not None and self.viewer.selected_entity != self._locked_entity:
            self.viewer.select_entity(self._locked_entity)

    def _build_ui(self):
        children = [R.UIDisplayText().Text("Teleop Controls")]
        if self._has_gripper:
            children.append(
                R.UISliderFloat()
                .Label("Gripper Opening")
                .Min(0.0)
                .Max(1.0)
                .Bind(self, "gripper_openness")
            )
        children.append(R.UIDisplayText().Text("Keys: h=help n=move g=gripper r=reset c=next q=quit"))
        self._ui_window = R.UIWindow().Label("Teleop").Pos(10, 10).Size(300, 120)
        for child in children:
            self._ui_window.append(child)

    def get_ui_windows(self):
        if self._ui_window is None:
            self._build_ui()
        return [self._ui_window]


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    """The robot to use. Any robot with ee_link_name and arm_joint_names is supported."""
    ee_link: Optional[str] = None
    """Override the end-effector link name. If not set, uses agent.ee_link_name."""
    obs_mode: str = "none"
    record_dir: str = "demos"
    """Directory to record demonstration data and optionally videos."""
    save_video: bool = False
    """Whether to save videos of the demonstrations after collecting them."""
    viewer_shader: str = _DEFAULT_SHADER
    """Shader for the viewer. 'default' is fast, 'rt' and 'rt-fast' are ray tracing. Defaults to 'default' on macOS."""
    video_saving_shader: str = _DEFAULT_SHADER
    """Shader for saved videos. 'default' is fast, 'rt' and 'rt-fast' are ray tracing. Defaults to 'default' on macOS."""
    step_size: float = 0.01
    """Translation step size for keyboard nudge controls."""


def parse_args() -> Args:
    return tyro.cli(Args)


def _get_robot_info(env, ee_link_override=None):
    """Extract robot introspection data from the environment agent."""
    agent = env.unwrapped.agent

    # EE link name
    ee_link_name = ee_link_override
    if ee_link_name is None:
        if hasattr(agent, "ee_link_name") and agent.ee_link_name:
            ee_link_name = agent.ee_link_name
        else:
            link_names = [l.name for l in agent.robot.links]
            print("ERROR: Robot does not define ee_link_name. Use --ee-link to specify one.")
            print(f"Available links: {link_names}")
            raise SystemExit(1)

    # Arm joint names
    if not hasattr(agent, "arm_joint_names") or not agent.arm_joint_names:
        print("ERROR: Robot does not define arm_joint_names. This script requires it.")
        raise SystemExit(1)
    arm_joint_names = agent.arm_joint_names

    # Arm joint indices (indices into the active joints array)
    arm_joint_indices = []
    for name in arm_joint_names:
        joint = agent.robot.active_joints_map[name]
        arm_joint_indices.append(joint.active_index[0].item())

    # Gripper detection
    has_gripper = (
        hasattr(agent, "gripper_joint_names")
        and agent.gripper_joint_names
        and len(agent.gripper_joint_names) > 0
    )
    gripper_joint_names = agent.gripper_joint_names if has_gripper else []

    # Gripper open/close values from joint limits
    gripper_open_values = []
    gripper_close_values = []
    if has_gripper:
        qlimits = agent.robot.get_qlimits()[0]  # (num_joints, 2), take first env
        for name in gripper_joint_names:
            joint = agent.robot.active_joints_map[name]
            idx = joint.active_index[0].item()
            lower = qlimits[idx, 0].item()
            upper = qlimits[idx, 1].item()
            gripper_open_values.append(upper)
            gripper_close_values.append(lower)

    # Action layout: CombinedController vs flat
    controller = agent.controller
    is_combined = isinstance(controller, CombinedController)
    action_mapping = controller.action_mapping if is_combined else None

    # URDF path
    urdf_path = format_path(str(agent.urdf_path))

    return dict(
        ee_link_name=ee_link_name,
        arm_joint_names=arm_joint_names,
        arm_joint_indices=arm_joint_indices,
        has_gripper=has_gripper,
        gripper_joint_names=gripper_joint_names,
        gripper_open_values=gripper_open_values,
        gripper_close_values=gripper_close_values,
        is_combined=is_combined,
        action_mapping=action_mapping,
        urdf_path=urdf_path,
    )


def _find_gizmo_link(agent, ee_link_name):
    """Find a visible link for the gizmo, walking up kinematic chain if needed.

    Returns (gizmo_link, ee_in_gizmo_offset) where ee_in_gizmo_offset is the
    static transform from the gizmo link frame to the EE link frame.
    """
    ee_link = sapien_utils.get_obj_by_name(agent.robot.links, ee_link_name)

    def _has_visual(link):
        """Check if a link has visual geometry."""
        try:
            entity = link._objs[0].entity
            for comp in entity.components:
                # Check for render body component (visual geometry)
                if hasattr(comp, "render_shapes") or "RenderBody" in type(comp).__name__:
                    return True
        except (IndexError, AttributeError):
            pass
        return False

    # Walk up the kinematic chain to find a link with visuals
    gizmo_link = ee_link
    while gizmo_link is not None and not _has_visual(gizmo_link):
        parent = gizmo_link.joint.parent_link
        if parent is None:
            break
        gizmo_link = parent

    # If we couldn't find any visual link, just use the EE link
    if gizmo_link is None:
        gizmo_link = ee_link

    # Compute static offset: ee pose expressed in gizmo link frame
    # Convert ManiSkill batched Pose to a sapien.Pose for use with the viewer gizmo
    ms_offset = gizmo_link.pose.inv() * ee_link.pose
    p = ms_offset.p[0].cpu().numpy()
    q = ms_offset.q[0].cpu().numpy()
    ee_in_gizmo = sapien.Pose(p=p, q=q)

    return gizmo_link, ee_in_gizmo


def _build_action(env, robot_info, arm_qpos, gripper_values):
    """Construct a full action vector for env.step().

    Args:
        env: the gym environment
        robot_info: dict from _get_robot_info
        arm_qpos: numpy array of target arm joint positions
        gripper_values: list of gripper joint target values (or empty if no gripper)
    """
    action_dim = env.action_space.shape[0]
    action = np.zeros(action_dim, dtype=np.float32)

    if robot_info["is_combined"]:
        mapping = robot_info["action_mapping"]
        # Fill arm action
        if "arm" in mapping:
            start, end = mapping["arm"]
            action[start:end] = arm_qpos
        # Fill gripper action(s)
        if robot_info["has_gripper"]:
            gripper_idx = 0
            for uid, (start, end) in mapping.items():
                if "gripper" in uid.lower() and "passive" not in uid.lower():
                    size = end - start
                    action[start:end] = gripper_values[gripper_idx:gripper_idx + size]
                    gripper_idx += size
    else:
        # Flat controller: arm joints first, then gripper
        n_arm = len(robot_info["arm_joint_names"])
        action[:n_arm] = arm_qpos
        if robot_info["has_gripper"] and len(gripper_values) > 0:
            action[n_arm:n_arm + len(gripper_values)] = gripper_values

    return action


def _compute_gripper_values(robot_info, openness):
    """Compute gripper joint values from a 0-1 openness fraction."""
    open_vals = robot_info["gripper_open_values"]
    close_vals = robot_info["gripper_close_values"]
    return [c + openness * (o - c) for o, c in zip(open_vals, close_vals)]


def _snap_gizmo_to_current(agent, gizmo_link, transform_window):
    """Snap the gizmo to the current gizmo link pose (e.g. after reset)."""
    p = gizmo_link.pose.p[0].cpu().numpy()
    q = gizmo_link.pose.q[0].cpu().numpy()
    current_pose = sapien.Pose(p=p, q=q)
    transform_window.gizmo_matrix = current_pose.to_transformation_matrix()
    transform_window.update_ghost_objects()


def solve(env: BaseEnv, robot_info: dict, step_size: float, teleop_plugin: TeleopPlugin):
    """Interactive teleoperation loop.

    Returns "quit" or "continue" to signal the outer loop.
    """
    base_env = env.unwrapped
    agent = base_env.agent
    ee_link_name = robot_info["ee_link_name"]

    # Setup IK solver
    kinematics = Kinematics(
        urdf_path=robot_info["urdf_path"],
        end_link_name=ee_link_name,
        articulation=agent.robot,
        active_joint_indices=torch.tensor(robot_info["arm_joint_indices"]),
    )

    viewer = base_env.render_human()

    # Find gizmo link and compute offset
    gizmo_link, ee_in_gizmo = _find_gizmo_link(agent, ee_link_name)
    gizmo_entity = gizmo_link._objs[0].entity

    # Lock selection to the gizmo entity — prevents accidentally clicking joints
    teleop_plugin.set_locked_entity(gizmo_entity)
    teleop_plugin.set_has_gripper(robot_info["has_gripper"])
    teleop_plugin.gripper_openness = 1.0  # start fully open
    viewer.select_entity(gizmo_entity)

    # Find transform window plugin
    transform_window = None
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin
            break

    # Store initial qpos for reset
    initial_qpos = agent.robot.get_qpos()[0].cpu().numpy().copy()

    # Track previous gripper openness to detect slider changes
    prev_gripper_openness = teleop_plugin.gripper_openness

    while True:
        transform_window.enabled = True
        base_env.render_human()
        execute_current_pose = False

        # --- Gripper slider: detect changes and actuate ---
        if robot_info["has_gripper"]:
            cur_openness = teleop_plugin.gripper_openness
            if abs(cur_openness - prev_gripper_openness) > 1e-4:
                gripper_vals = _compute_gripper_values(robot_info, cur_openness)
                qpos = agent.robot.get_qpos()[0].cpu().numpy()
                arm_qpos = np.array([qpos[i] for i in robot_info["arm_joint_indices"]])
                for _ in range(4):
                    action = _build_action(env, robot_info, arm_qpos, gripper_vals)
                    env.step(action)
                    base_env.render_human()
                prev_gripper_openness = cur_openness

        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            u: move EE gizmo up
            j: move EE gizmo down
            arrow keys: move EE gizmo in X/Y directions
            n: execute IK motion to move robot to gizmo target pose
            g: toggle gripper open/close (if robot has one)
            r: reset robot to initial pose
            c: save episode and start a new one
            q: save and quit
            Gripper slider in the Teleop panel for fine control.""")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("r"):
            # Reset robot to initial pose
            arm_qpos = np.array([initial_qpos[i] for i in robot_info["arm_joint_indices"]])
            if robot_info["has_gripper"]:
                gripper_vals = _compute_gripper_values(robot_info, 1.0)
                teleop_plugin.gripper_openness = 1.0
                prev_gripper_openness = 1.0
            else:
                gripper_vals = []
            for _ in range(20):
                action = _build_action(env, robot_info, arm_qpos, gripper_vals)
                env.step(action)
                base_env.render_human()
            # Snap gizmo to new EE position
            _snap_gizmo_to_current(agent, gizmo_link, transform_window)
            print("Reset to initial pose")
        elif viewer.window.key_press("g") and robot_info["has_gripper"]:
            # Toggle gripper via key (snap to fully open or fully closed)
            if teleop_plugin.gripper_openness > 0.5:
                teleop_plugin.gripper_openness = 0.0
            else:
                teleop_plugin.gripper_openness = 1.0
            gripper_vals = _compute_gripper_values(robot_info, teleop_plugin.gripper_openness)
            prev_gripper_openness = teleop_plugin.gripper_openness

            qpos = agent.robot.get_qpos()[0].cpu().numpy()
            arm_qpos = np.array([qpos[i] for i in robot_info["arm_joint_indices"]])
            for _ in range(6):
                action = _build_action(env, robot_info, arm_qpos, gripper_vals)
                env.step(action)
                base_env.render_human()
            print(f"Gripper {'opened' if teleop_plugin.gripper_openness > 0.5 else 'closed'}")
        elif viewer.window.key_press("u"):
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -step_size])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +step_size])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[+step_size, 0, 0])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[-step_size, 0, 0])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[0, -step_size, 0])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            transform_window.gizmo_matrix = (
                transform_window._gizmo_pose * sapien.Pose(p=[0, +step_size, 0])
            ).to_transformation_matrix()
            transform_window.update_ghost_objects()

        if execute_current_pose:
            # Compute target EE pose from gizmo pose + offset (world frame)
            gizmo_pose = transform_window._gizmo_pose
            target_ee_pose_world = gizmo_pose * ee_in_gizmo

            # Transform from world frame to robot base frame for Pinocchio IK
            robot_pose = agent.robot.pose
            robot_pose_sp = sapien.Pose(
                p=robot_pose.p[0].cpu().numpy(),
                q=robot_pose.q[0].cpu().numpy(),
            )
            target_ee_pose_base = robot_pose_sp.inv() * target_ee_pose_world

            # Run IK
            q0 = agent.robot.get_qpos()
            target_pose_ms = MSPose.create_from_pq(
                p=torch.tensor([[target_ee_pose_base.p[0], target_ee_pose_base.p[1], target_ee_pose_base.p[2]]], dtype=torch.float32),
                q=torch.tensor([[target_ee_pose_base.q[0], target_ee_pose_base.q[1], target_ee_pose_base.q[2], target_ee_pose_base.q[3]]], dtype=torch.float32),
            )
            ik_result = kinematics.compute_ik(
                pose=target_pose_ms,
                q0=q0,
            )

            if ik_result is None:
                print("IK failed — try a closer target")
            else:
                target_arm_qpos = ik_result[0].cpu().numpy()

                # Get current arm qpos
                current_qpos = q0[0].cpu().numpy()
                current_arm_qpos = np.array([current_qpos[i] for i in robot_info["arm_joint_indices"]])

                # Current gripper from slider
                if robot_info["has_gripper"]:
                    gripper_vals = _compute_gripper_values(robot_info, teleop_plugin.gripper_openness)
                else:
                    gripper_vals = []

                # Interpolate from current to target
                max_delta = np.max(np.abs(target_arm_qpos - current_arm_qpos))
                n_steps = max(1, min(100, int(max_delta / 0.02)))

                for step_i in range(1, n_steps + 1):
                    t = step_i / n_steps
                    interp_arm = current_arm_qpos + t * (target_arm_qpos - current_arm_qpos)
                    action = _build_action(env, robot_info, interp_arm, gripper_vals)
                    env.step(action)
                    base_env.render_human()

                print(f"Moved to target in {n_steps} steps")

            execute_current_pose = False


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
        viewer_camera_configs=dict(shader_pack=args.viewer_shader),
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system",
    )

    seed = 0
    num_trajs = 0
    env.reset(seed=seed)

    # Extract robot info once after first reset
    robot_info = _get_robot_info(env, ee_link_override=args.ee_link)
    print(f"Robot: {args.robot_uid}")
    print(f"EE link: {robot_info['ee_link_name']}")
    print(f"Arm joints: {robot_info['arm_joint_names']}")
    print(f"Has gripper: {robot_info['has_gripper']}")
    if robot_info["is_combined"]:
        print(f"Action mapping: {robot_info['action_mapping']}")
    print("Press 'h' in the viewer for controls help.\n")

    # Create teleop plugin (will be registered with viewer on first render)
    teleop_plugin = TeleopPlugin()

    # Do a first render to get the viewer, then register the plugin
    base_env = env.unwrapped
    viewer = base_env.render_human()
    teleop_plugin.init(viewer)
    viewer.plugins.append(teleop_plugin)

    while True:
        print(f"Collecting trajectory {num_trajs + 1}, seed={seed}")
        code = solve(env, robot_info, step_size=args.step_size, teleop_plugin=teleop_plugin)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue

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
            video_fps=30,
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


if __name__ == "__main__":
    main(parse_args())
