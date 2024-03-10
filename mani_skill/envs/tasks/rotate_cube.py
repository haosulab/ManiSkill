"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill2 tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self.reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill2. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from collections import OrderedDict
from typing import Any, Dict, Union, Tuple

import numpy as np
import sapien.pysapien.physx
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.xmate3.xmate3 import Xmate3Robotiq
from mani_skill.agents.robots.trifingerpro.trifingerpro import TriFingerPro
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors, ActorBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("RotateCubeEnv-v1", max_episode_steps=250)
class RotateCubeEnv(BaseEnv):
    """
    Task Description
    ----------------
    A simple task where the objective is to push and move a cube to a goal region in front of it

    Randomizations
    --------------
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]

    Success Conditions
    ------------------
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.

    Visualization: TODO: ADD LINK HERE
    """

    """
    Modified from https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/trifinger.py
    https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/Trifinger.yaml
    """

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch", "trifingerpro"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch, TriFingerPro]

    # Specify default simulation/gpu memory configurations.
    sim_cfg = SimConfig(
        gpu_memory_cfg=GPUMemoryConfig(
            found_lost_pairs_capacity=2 ** 25, max_rigid_patch_count=2 ** 18
        )
    )

    # set some commonly used values
    goal_radius = 0.02
    cube_half_size = 0.02

    # radius of the area
    ARENA_RADIUS = 0.195
    size = 0.065  # m
    max_len = 0.065
    # 3D radius of the cuboid
    radius_3d = max_len * np.sqrt(3) / 2
    # compute distance from wall to the center
    max_com_distance_to_center = ARENA_RADIUS - radius_3d
    # minimum and maximum height for spawning the object
    min_height = 0.065 / 2
    max_height = 0.1

    def __init__(self, *args, robot_uids="trifingerpro", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.prev_norms = None

    @property
    def _sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = look_at(eye=(0.7, 0.0, 0.7), target=(0.0, 0.0, 0.0))
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    @property
    def _human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = look_at((0.7, 0.0, 0.7), (0.0, 0.0, 0.0))
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _load_scene(self):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.

        loader1 = self._scene.create_urdf_loader()
        loader1.fix_root_link = True
        loader1.name = "table"
        urdf_path = f"{PACKAGE_ASSET_DIR}/trifinger/table_without_border.urdf"
        table: Articulation = loader1.load(urdf_path)

        # builder: ActorBuilder = self._scene.create_actor_builder()
        # cube_file_path = "/home/chenbao/Desktop/projects/ManiSkill3/mani_skill2/assets/trifinger/objects/meshes/cube_multicolor.obj"
        # cube_material = sapien.pysapien.physx.PhysxMaterial(static_friction=1, dynamic_friction=3, restitution=0)
        # builder.add_nonconvex_collision_from_file(filename=cube_file_path, scale=(0.065, 0.065, 0.065))#, material=cube_material)
        # builder.add_visual_from_file(filename=cube_file_path, scale=(0.065, 0.065, 0.065))
        # builder._mass = 1
        # self.obj = builder.build("cube")
        #
        # builder: ActorBuilder = self._scene.create_actor_builder()
        # builder.add_visual_from_file(filename=cube_file_path, scale=(0.065, 0.065, 0.065))
        # self.obj_goal = builder.build_kinematic("cube_goal")

        builder: ActorBuilder = self._scene.create_actor_builder()
        high_table_boundary_file_name = f"{PACKAGE_ASSET_DIR}/trifinger/robot_properties_fingers/meshes/high_table_boundary.stl"
        builder.add_nonconvex_collision_from_file(filename=high_table_boundary_file_name, scale=[1, 1, 1], material=None)
        builder.add_visual_from_file(filename=high_table_boundary_file_name)
        table_boundary: Actor = builder.build_static("table2")

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        self.obj = actors.build_cube(
            self._scene,
            half_size=self.size / 2,
            color=np.array([169, 42, 12, 255]) / 255,
            name="cube",
            body_type="dynamic",
            add_collision=True,
        )

        self.obj_goal = actors.build_cube(
            self._scene,
            half_size=self.size / 2,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube_goal",
            body_type="kinematic",
            add_collision=False
        )

        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        # self.goal_region = actors.build_red_white_target(
        #     self._scene,
        #     radius=self.goal_radius,
        #     thickness=1e-5,
        #     name="goal_region",
        #     add_collision=False,
        #     body_type="kinematic",
        # )

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        # self._hidden_objects.append(self.goal_region)

    def _initialize_actors(self, env_idx: torch.Tensor):
        # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            # the initialization functions where you as a user place all the objects and initialize their properties
            # are designed to support partial resets, where you generate initial state for a subset of the environments.
            # this is done by using the env_idx variable, which also tells you the batch size
            b = len(env_idx)
            # when using scene builders, you must always call .initialize on them so they can set the correct poses of objects in the prebuilt scene
            # note that the table scene is built such that z=0 is the surface of the table.
            # self.table_scene.initialize()

            # here we write some randomization code that randomizes the x, y position of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
            xyz = torch.zeros((b, 3))
            # xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.size / 2 + 0.005
            # we can then create a pose object using Pose.create_from_pq to then set the cube pose with. Note that even though our quaternion
            # is not batched, Pose.create_from_pq will automatically batch p or q accordingly
            # furthermore, notice how here we do not even using env_idx as a variable to say set the pose for objects in desired
            # environments. This is because internally any calls to set data on the GPU buffer (e.g. set_pose, set_linear_velocity etc.)
            # automatically are masked so that you can only set data on objects in environments that are meant to be initialized
            obj_pose = Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0])
            self.obj.set_pose(obj_pose)

            # here we set the location of that red/white target (the goal region). In particular here, we set the position to be in front of the cube
            # and we further rotate 90 degrees on the y-axis to make the target object face up
            pos, orn = self._sample_object_goal_poses(env_idx, difficulty=1)
            # set a little bit above 0 so the target is sitting on the table
            # target_region_xyz[..., 2] = 1e-3
            self.obj_goal.set_pose(
                Pose.create_from_pq(
                    p=pos,
                    q=orn,
                )
            )
            self.prev_norms = None

    def _initialize_episode(self, env_idx: torch.Tensor):
        self._initialize_actors(env_idx)
        self._initialize_agent(env_idx)

    def _sample_object_goal_poses(self, env_idx: torch.Tensor, difficulty: int):
        """Sample goal poses for the cube and sets them into the desired goal pose buffer.

        Args:
            instances: A tensor constraining indices of environment instances to reset.
            difficulty: Difficulty level. The higher, the more difficult is the goal.

        Possible levels are:
            - -1:  Random goal position on the table, including yaw orientation.
            - 1: Random goal position on the table, no orientation.
            - 2: Fixed goal position in the air with x,y = 0.  No orientation.
            - 3: Random goal position in the air, no orientation.
            - 4: Random goal pose in the air, including orientation.
        """
        # number of samples to generate
        b = len(env_idx)
        # sample poses based on task difficulty
        if difficulty == -1:
            # For initialization
            pos_x, pos_y = random_xy(b, self.max_com_distance_to_center, self.device)
            pos_z = self.size / 2
            orientation = random_roll_orientation(b, self.device)
        elif difficulty == 1:
            # Random goal position on the table, no orientation.
            pos_x, pos_y = random_xy(b, self.max_com_distance_to_center, self.device)
            pos_z = self.size / 2
            orientation = default_orientation(b, self.device)
        elif difficulty == 2:
            # Fixed goal position in the air with x,y = 0.  No orientation.
            pos_x, pos_y = 0.0, 0.0
            pos_z = self.min_height + 0.05
            orientation = default_orientation(b, self.device)
        elif difficulty == 3:
            # Random goal position in the air, no orientation.
            pos_x, pos_y = random_xy(b, self.max_com_distance_to_center, self.device)
            pos_z = random_z(b, self.min_height, self.max_height, self.device)
            orientation = default_orientation(b, self.device)
        elif difficulty == 4:
            # Random goal pose in the air, including orientation.
            # Note: Set minimum height such that the cube does not intersect with the
            #       ground in any orientation
            max_goal_radius = self.max_com_distance_to_center
            max_height = self.max_height
            orientation = random_orientation(b, self.device)

            # pick x, y, z according to the maximum height / radius at the current point
            # in the cirriculum
            pos_x, pos_y = random_xy(b, max_goal_radius, self.device)
            pos_z = random_z(b, self.radius_3d, max_height, self.device)
        else:
            msg = f"Invalid difficulty index for task: {difficulty}."
            raise ValueError(msg)

        pos_tensor = torch.zeros((b, 3), dtype=torch.float, device=self.device)
        pos_tensor[:, 0] = pos_x
        pos_tensor[:, 1] = pos_y
        pos_tensor[:, 2] = pos_z
        return pos_tensor, orientation

    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position)
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.obj_goal.pose.p[..., :2], axis=1
            )
            < self.goal_radius / 10
        )

        return {
            "success": is_obj_placed,
        }
        obj_p = self.obj.pose.p
        goal_p = self.obj_goal.pose.p
        obj_q = self.obj.pose.q
        goal_q = self.obj_goal.pose.q
        obj_lin_vel = self.obj.linear_velocity
        obj_ang_vel = self.obj.angular_velocity

        is_obj_pos_close_to_goal = (
                torch.linalg.norm(obj_p - goal_p, axis=1) < 0.0005
        )

        is_obj_state_stable = torch.linalg.norm(obj_lin_vel, axis=1) < 0.01
        is_obj_state_stable &= torch.linalg.norm(obj_ang_vel, axis=1) < 0.01
        # print("is_obj_pos_close_to_goal", is_obj_pos_close_to_goal)
        is_obj_q_close_to_goal = (
                quat_diff_rad(obj_q, goal_q) < 0.1
        )

        is_success = is_obj_pos_close_to_goal & is_obj_state_stable & is_obj_q_close_to_goal

        return {
            "success": is_success,
        }

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]

            init_qpos = torch.zeros((b, dof))
            # # set root joint qpos to avoid robot-object collision after reset
            # init_qpos[:, self.agent.root_joint_indices] = torch.tensor(
            #     [0.7, -0.7, -0.7]
            # )
            # init_qpos += torch.randn((b, dof)) * self.robot_init_qpos_noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, self.size / 2 + 0.022]), torch.tensor([1, 0, 0, 0])
                )
            )

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = OrderedDict(
            goal_pos=self.obj_goal.pose.p,
            goal_q=self.obj_goal.pose.q,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                obj_p=self.obj.pose.p,
                obj_q=self.obj.pose.q,
            )
            pass
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)

        obj_pos = self.obj.pose.p
        obj_q = self.obj.pose.q
        goal_pos = self.obj_goal.pose.p
        goal_q = self.obj_goal.pose.q

        finger_move_penalty_weight = -0.1
        finger_reach_object_weight = -50
        dt = self.physx_system.timestep

        # Reward penalising finger movement

        tip_poses = self.agent.tip_poses()
        # shape (N, 3 + 4, 3 fingers)

        fingertip_vel = self.agent.tip_velocities()
        # shape (N, 3(linear velocity), 3 fingers)

        finger_movement_penalty = finger_move_penalty_weight * fingertip_vel.pow(2).view(-1, 9).sum(dim=-1)

        # Reward for finger reaching the object

        # distance from each finger to the centroid of the object, shape (N, 3).
        curr_norms = torch.stack([
            torch.norm(tip_poses[:, :3, i] - obj_pos, p=2, dim=-1)
            for i in range(3)
        ], dim=-1)
        # distance from each finger to the centroid of the object in the last timestep, shape (N, 3).

        ft_sched_val = 1.0  # if ft_sched_start <= env_steps_count <= ft_sched_end else 0.0
        if self.prev_norms is not None:
            finger_reach_object_reward = finger_reach_object_weight * ft_sched_val * (curr_norms - self.prev_norms).sum(dim=-1)
        else:
            finger_reach_object_reward = torch.zeros_like(curr_norms.sum(dim=-1))
        self.prev_norms = curr_norms

        # Reward for object distance
        object_dist = torch.norm(obj_pos - goal_pos, p=2, dim=-1)
        # object_dist_reward = object_dist_weight * dt * lgsk_kernel(object_dist, scale=50., eps=2.)

        object_dist_reward = (1 - torch.tanh(5 * object_dist))

        # Reward for object rotation

        # extract quaternion orientation
        angles = quat_diff_rad(obj_q, goal_q)
        object_rot_reward = dt / (3. * torch.abs(angles) + 0.01)

        pose_reward = object_dist_reward + object_rot_reward

        # total_reward = finger_movement_penalty + finger_reach_object_reward + pose_reward
        total_reward = pose_reward
        # total_reward = torch.clamp(total_reward, min=-50)
        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1


@torch.jit.script
def random_xy(num: int, max_com_distance_to_center: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
    # sample radius of circle
    radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    radius *= max_com_distance_to_center
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    # x,y-position of the cube
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y


@torch.jit.script
def random_z(num: int, min_height: float, max_height: float, device: torch.device) -> torch.Tensor:
    """Returns sampled height of the goal object."""
    z = torch.rand(num, dtype=torch.float, device=device)
    z = (max_height - min_height) * z + min_height

    return z


@torch.jit.script
def default_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns identity rotation transform."""
    quat = torch.zeros((num, 4,), dtype=torch.float, device=device)
    quat[..., 0] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.
    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4,), dtype=torch.float, device=device)
    # normalize the quaternion
    quat = torch.nn.functional.normalize(quat, p=2., dim=-1, eps=1e-12)

    return quat


# @torch.jit.script
# def random_orientation_within_angle(num: int, device: str, base: torch.Tensor, max_angle: float):
#     """ Generates random quaternions within max_angle of base
#     Ref: https://math.stackexchange.com/a/3448434
#     """
#     quat = torch.zeros((num, 4,), dtype=torch.float, device=device)
#
#     rand = torch.rand((num, 3), dtype=torch.float, device=device)
#
#     c = torch.cos(rand[:, 0] * max_angle)
#     n = torch.sqrt((1. - c) / 2.)
#
#     quat[:, 3] = torch.sqrt((1 + c) / 2.)
#     quat[:, 2] = (rand[:, 1] * 2. - 1.) * n
#     quat[:, 0] = (torch.sqrt(1 - quat[:, 2] ** 2.) * torch.cos(2 * np.pi * rand[:, 2])) * n
#     quat[:, 1] = (torch.sqrt(1 - quat[:, 2] ** 2.) * torch.sin(2 * np.pi * rand[:, 2])) * n
#
#     # floating point errors can cause it to  be slightly off, re-normalise
#     quat = torch.nn.functional.normalize(quat, p=2., dim=-1, eps=1e-12)
#
#     return quat_mul(quat, base)


# @torch.jit.script
# def random_angular_vel(num: int, device: str, magnitude_stdev: float) -> torch.Tensor:
#     """Samples a random angular velocity with standard deviation `magnitude_stdev`"""
#
#     axis = torch.randn((num, 3,), dtype=torch.float, device=device)
#     axis /= torch.norm(axis, p=2, dim=-1).view(-1, 1)
#     magnitude = torch.randn((num, 1,), dtype=torch.float, device=device)
#     magnitude *= magnitude_stdev
#     return magnitude * axis

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def random_roll_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation around z-axis."""
    roll = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = torch.zeros(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)


#
#
# @torch.jit.script
# def get_euler_xyz(q):
#     qx, qy, qz, qw = 0, 1, 2, 3
#     # roll (x-axis rotation)
#     sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
#     cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
#                 q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
#     roll = torch.atan2(sinr_cosp, cosr_cosp)
#
#     # pitch (y-axis rotation)
#     sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
#     pitch = torch.where(torch.abs(sinp) >= 1, copysign(
#         np.pi / 2.0, sinp), torch.asin(sinp))
#
#     # yaw (z-axis rotation)
#     siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
#     cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
#                 q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
#     yaw = torch.atan2(siny_cosp, cosy_cosp)
#
#     return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)
#
#

# @torch.jit.script
# def copysign(a, b):
#     # type: (float, Tensor) -> Tensor
#     a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
#     return torch.abs(a) * torch.sign(b)

@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:4]), dim=-1).view(shape)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    # Normalize the quaternions
    a = a / torch.norm(a, dim=1, keepdim=True)
    b = b / torch.norm(b, dim=1, keepdim=True)

    # Compute the dot product between the quaternions
    dot_product = torch.sum(a * b, dim=1)

    # Clamp the dot product to the range [-1, 1] to avoid numerical instability
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the angle difference in radians
    angle_diff = 2 * torch.acos(torch.abs(dot_product))

    return angle_diff


@torch.jit.script
def lgsk_kernel(x: torch.Tensor, scale: float = 50.0, eps: float = 2) -> torch.Tensor:
    """Defines logistic kernel function to bound input to [-0.25, 0)

    Ref: https://arxiv.org/abs/1901.08652 (page 15)

    Args:
        x: Input tensor.
        scale: Scaling of the kernel function (controls how wide the 'bell' shape is')
        eps: Controls how 'tall' the 'bell' shape is.

    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    return 1.0 / (scaled.exp() + eps + (-scaled).exp())
