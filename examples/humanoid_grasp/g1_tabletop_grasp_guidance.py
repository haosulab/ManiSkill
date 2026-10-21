"""A minimal, self-contained G1 humanoid tabletop grasp environment that
demonstrates the reward / observation / reset recommendations discussed in
https://github.com/mani-skill/ManiSkill/issues/1465.

It is intended as a *starting template* to copy into your own task, not as a
new benchmark task. Run it on CPU (no rendering needed):

    python examples/humanoid_grasp/g1_tabletop_grasp_guidance.py

Design notes (why each term looks the way it does):

1. **Reaching reward is gated by ``1 - is_grasped``** and computed with a
   ``tanh`` kernel, mirroring ``PickSingleYCBEnv`` (mani_skill/envs/tasks/tabletop/pick_single_ycb.py).
   Once the hand is at the object, extra hovering earns almost nothing, so the
   policy must progress to the next (grasp) term to collect more reward. This
   directly addresses the "policy hovers near the object forever" problem.

2. **``is_grasped`` uses the contact-force based ``right_hand_is_grasping``
   check with a slightly raised ``min_force``** (1.5 N instead of the 0.5 N
   default) so that lightly brushing the object does not count as a grasp.
   Reward only pays out when contact forces are large enough to actually hold
   the object, which is what enables lifting.

3. **Lift / hold terms are all multiplicative gates on ``is_grasped``**, so no
   reward can be farmed by knocking the object around: lifting only pays while
   grasped, static-hold only pays while grasped *and* lifted.

4. **Observations are root-relative and palm-relative** (plus gripper width and
   the ``is_grasped`` flag), which removes the world-frame dependence and makes
   the policy far easier to transfer to new object/table placements.

5. **Reset starts from an open-handed pregrasp-style keyframe** so that the
   very first reward gradient the policy sees is about closing the fingers on
   the object rather than discovering a whole reaching curriculum from a
   standing-by-the-side pose.
"""

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import build_box, build_cube
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


@register_env("G1TabletopGraspGuidance-v1", max_episode_steps=100)
class G1TabletopGraspGuidanceEnv(BaseEnv):
    """
    **Task Description:**
    Control the Unitree G1 (simplified upper body, right arm only used by the
    reward) to grasp a small cube that rests on a table in front of it and hold
    it lifted above its initial height.

    This env exists as a runnable reference for the reward/obs/reset design
    discussed in issue #1465.
    """

    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body"]
    agent: UnitreeG1UpperBody

    # reward weights
    w_reach = 1.0
    w_grasp = 1.0
    w_lift = 1.0
    w_hold = 0.5
    w_success = 3.0
    # reaching kernel scale (larger = sharper gradient near the object)
    reaching_scale = 8.0
    # minimum pairwise contact force norm (N) for a finger to count as grasping
    grasp_min_force = 1.5
    # object must be lifted this much (m) above its initial height to count
    lift_height_threshold = 0.08
    # object velocities below this (rad/s, m/s) count as "static"
    static_lin_vel_thresh = 0.1
    static_ang_vel_thresh = 1.0

    def __init__(self, *args, robot_uids="unitree_g1_simplified_upper_body", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**22),
            scene_config=SceneConfig(contact_offset=0.01),
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 0.8, 1.3], [0.3, 0, 0.8])
        return CameraConfig("render_camera", pose, 512, 512, np.pi / 2, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.755]))

    def _load_scene(self, options: dict):
        # a simple static table in front of the robot
        self.table = build_box(
            scene=self.scene,
            half_sizes=[0.25, 0.4, 0.35],
            color=[0.6, 0.4, 0.2, 1],
            name="table",
            body_type="static",
            initial_pose=sapien.Pose(p=[0.35, 0, 0.35]),
        )
        # a small cube object to grasp (jenga-block-like, deliberately small)
        self.cube_half_size = 0.025
        self.obj = build_cube(
            scene=self.scene,
            half_size=self.cube_half_size,
            color=[0.1, 0.5, 0.9, 1],
            name="grasp_object",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.28, 0, 0.7 + self.cube_half_size]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # small xy randomization of the object on the table
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = 0.28 + torch.rand(b) * 0.1 - 0.05
            xyz[:, 1] = torch.rand(b) * 0.1 - 0.05
            xyz[:, 2] = 0.7 + self.cube_half_size
            self.obj.set_pose(Pose.create_from_pq(p=xyz))

            # Pregrasp-style reset: right arm in front of the object with an
            # open hand (pose found with a small grid search minimizing the
            # right_tcp-to-object distance), so the early training gradient is
            # about closing the fingers on the object, not about discovering
            # reaching from scratch.
            qpos = np.zeros(self.agent.robot.dof)
            aj = self.agent.robot.active_joints_map
            qpos[aj["right_shoulder_pitch_joint"].active_index[0]] = -0.4
            qpos[aj["right_shoulder_roll_joint"].active_index[0]] = -0.2
            qpos[aj["right_shoulder_yaw_joint"].active_index[0]] = -0.4
            qpos[aj["right_elbow_pitch_joint"].active_index[0]] = -1.6
            self.agent.reset(qpos)
            self.agent.robot.set_root_pose(sapien.Pose(p=[0, 0, 0.755]))

            self.initial_object_z = self.obj.pose.p[:, 2].clone()

    def evaluate(self):
        is_grasped = self.agent.right_hand_is_grasping(
            self.obj, min_force=self.grasp_min_force
        )
        obj_height = self.obj.pose.p[:, 2]
        is_lifted = obj_height > self.initial_object_z + self.lift_height_threshold
        return dict(
            is_grasped=is_grasped,
            is_lifted=is_lifted,
            success=torch.logical_and(is_grasped, is_lifted),
        )

    def _get_obs_extra(self, info: dict):
        # root-relative and palm-relative object pose, gripper aperture proxy,
        # and the grasp flag. All quantities are already in frames the policy
        # can relate to each other, which helps generalization across
        # object/table placements.
        root_inv = self.agent.robot.root.pose.inv()
        ee_pose = (root_inv * self.agent.right_tcp.pose).raw_pose
        object_pose = (root_inv * self.obj.pose).raw_pose
        object_pose_palm = (
            self.agent.right_tcp.pose.inv() * self.obj.pose
        ).raw_pose
        gripper_width = self.agent.right_hand_dist_to_open_grasp().unsqueeze(-1)
        obs = dict(
            ee_pose=ee_pose,
            gripper_width=gripper_width,
            object_pose=object_pose,
            object_pose_palm=object_pose_palm,
            is_grasped=info["is_grasped"].float().unsqueeze(-1),
        )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        is_grasped = info["is_grasped"].float()

        # 1) reaching reward, gated so it saturates the moment the hand is at
        #    the object AND disappears once grasped (no more hovering value)
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.right_tcp.pose.p, axis=1
        )
        reaching_reward = 1.0 - torch.tanh(self.reaching_scale * tcp_to_obj_dist)
        reward = self.w_reach * (1.0 - is_grasped) * reaching_reward

        # 2) contact-force based grasp reward
        reward = reward + self.w_grasp * is_grasped

        # 3) lift progress, only paid while grasped
        lift_progress = torch.clamp(
            (self.obj.pose.p[:, 2] - self.initial_object_z)
            / self.lift_height_threshold,
            0.0,
            1.0,
        )
        reward = reward + self.w_lift * is_grasped * lift_progress

        # 4) stable hold: grasp + lifted + object not moving
        lin_vel = torch.linalg.norm(self.obj.linear_velocity, dim=1)
        ang_vel = torch.linalg.norm(self.obj.angular_velocity, dim=1)
        static = (1.0 - torch.tanh(5.0 * lin_vel)) * (1.0 - torch.tanh(2.0 * ang_vel))
        reward = reward + self.w_hold * is_grasped * info["is_lifted"].float() * static

        # 5) success bonus
        reward = reward + self.w_success * info["success"].float()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / (
            self.w_reach + self.w_grasp + self.w_lift + self.w_hold + self.w_success
        )


if __name__ == "__main__":
    import gymnasium as gym

    import mani_skill.envs  # noqa: F401  (registers the envs)

    env = gym.make(
        "G1TabletopGraspGuidance-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        sim_backend="cpu",
        render_backend="none",  # no rendering needed for the smoke test
        num_envs=1,
    )
    obs, _ = env.reset(seed=0)
    def _shape(v):
        return tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
    print("obs keys:", {k: _shape(v) for k, v in obs.items()})
    # hold still with zero actions first so we can see the pregrasp reward
    for i in range(20):
        obs, reward, terminated, truncated, info = env.step(
            torch.zeros(1, env.action_space.shape[-1])
        )
        if i % 5 == 0:
            print(
                f"step {i}: reward={reward.mean().item():.4f} "
                f"is_grasped={info['is_grasped'].float().mean().item():.2f} "
                f"success={info['success'].float().mean().item():.2f}"
            )
    e = env.unwrapped
    print(
        "tcp-to-obj dist at end:",
        torch.linalg.norm(e.obj.pose.p - e.agent.right_tcp.pose.p).item(),
    )
    env.close()
    print("smoke test passed")
