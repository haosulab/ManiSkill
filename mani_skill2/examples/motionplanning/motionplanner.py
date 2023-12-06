import mplib
import numpy as np
import sapien

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.envs.sapien_env import BaseEnv

OPEN = 1
CLOSED = -1


class PandaArmMotionPlanningSolver:
    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        self.env = env
        self.env_agent: BaseAgent = self.env.unwrapped
        self.robot = self.env.agent.robot
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.planner = self.setup_planner()
        self.control_mode = self.env.unwrapped.control_mode
        self.base_pose = base_pose
        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.gripper_state = OPEN
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            self.grasp_pose_visual = build_panda_gripper_grasp_pose_visual(
                env.unwrapped._scene
            )
            self.grasp_pose_visual.set_pose(env.unwrapped.agent.tcp.entity_pose)
        self.elapsed_steps = 0

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.env.unwrapped.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.env.unwrapped.render_human()

    def setup_planner(self):
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env.agent.urdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="panda_hand_tcp",
            joint_vel_limits=np.ones(7) * self.joint_acc_limits,
            joint_acc_limits=np.ones(7) * self.joint_vel_limits,
        )
        return planner

    def follow_path(self, result):
        n_step = result["position"].shape[0]
        for i in range(n_step):
            qpos = result["position"][i]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][i]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.env.unwrapped.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_RRTConnect(self, pose):
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p - self.base_pose.p, q=pose.q)
        result = self.planner.plan(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos(),
            time_step=self.env.unwrapped.control_timestep,
        )
        if result["status"] != "Success":
            print(result["status"])
            self.render_wait()
            return -1
        self.render_wait()
        return self.follow_path(result)

    def move_to_pose_with_screw(self, pose: sapien.Pose):
        # try screw two times before giving up
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.set_pose(pose)
        pose = sapien.Pose(p=pose.p - self.base_pose.p, q=pose.q)
        result = self.planner.plan_screw(
            np.concatenate([pose.p, pose.q]),
            self.robot.get_qpos(),
            time_step=self.env.unwrapped.control_timestep,
        )
        if result["status"] != "Success":
            result = self.planner.plan(
                np.concatenate([pose.p, pose.q]),
                self.robot.get_qpos(),
                time_step=self.env.unwrapped.control_timestep,
            )
            if result["status"] != "Success":
                print(result["status"])
                self.render_wait()
                return -1
        self.render_wait()
        return self.follow_path(result)

    def open_gripper(self):
        self.gripper_state = OPEN
        qpos = self.robot.get_qpos()[:-2]
        for i in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.env.unwrapped.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6):
        self.gripper_state = CLOSED
        qpos = self.robot.get_qpos()[:-2]
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.env.unwrapped.render_human()
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.grasp_pose_visual is not None:
            self.grasp_pose_visual.remove_from_scene()


from transforms3d import quaternions


def build_panda_gripper_grasp_pose_visual(scene: sapien.Scene):
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01
    grasp_width = 0.05
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )
    grasp_pose_visual = builder.build_kinematic(name="grasp_pose_visual")
    return grasp_pose_visual
