from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the cube's half-size is resampled from `cube_half_size_range` on every reconfiguration
- the target goal position is fixed at `fixed_goal_pos` and is not marked by a visible object

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)

**ReCAST modifications to upstream ManiSkill:**
This task is customized for ReCAST data generation. Each behaviour is controlled by a
class attribute and can be reverted to the upstream default:

- `cube_half_size_range`: the cube is resized every reconfiguration. The range is
  bounded so the Panda gripper (0.08m maximum opening) can always close on the cube.
  Set to ``None`` to use the fixed size from `PICK_CUBE_CONFIGS`.
- `fixed_goal_pos`: the goal is the same every episode. Set to ``None`` to restore the
  upstream randomized goal.
- `render_goal_site`: the goal marker is built without any visual body, so the green
  sphere never appears in sensor observations *or* human-render videos. The
  ``goal_site`` actor still exists and still carries the goal pose, so motion planning
  solutions, reward terms and simulation-state tests are unaffected. Set to ``True``
  to restore the upstream green sphere.
- `sensor_camera_resolution`: resolution of `base_camera`.
- the default robot is `panda_wristcam`, which adds a wrist-mounted `hand_camera`
  alongside `base_camera`. Pass ``robot_uids="panda"`` for the upstream single-camera setup.

Note that the cube is only resized when the environment reconfigures. Pass
``reconfiguration_freq=1`` to `gym.make` (or ``options=dict(reconfigure=True)`` to
`env.reset`) to get a new cube size every episode. A single size is sampled per
reconfiguration and shared by all parallel sub-scenes.
"""


@register_env("PickCube-v1", max_episode_steps=50)
class PickCubeEnv(BaseEnv):

    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "panda_wristcam",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    # --- ReCAST customization (see the class docstring to restore upstream behaviour) ---
    # Bounded so the Panda gripper (0.08m max opening) can always grasp the cube.
    cube_half_size_range = (0.015, 0.028)
    fixed_goal_pos = (0.0, 0.0, 0.20)
    render_goal_site = False
    sensor_camera_resolution = (480, 480)

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]

        # The wrist camera is defined by the agent, not the task, so its resolution is
        # defaulted here to match base_camera. Explicit caller settings take precedence.
        if robot_uids == "panda_wristcam":
            sensor_configs = dict(kwargs.pop("sensor_configs", None) or {})
            width, height = self.sensor_camera_resolution
            sensor_configs.setdefault("hand_camera", dict(width=width, height=height))
            kwargs["sensor_configs"] = sensor_configs

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        width, height = self.sensor_camera_resolution
        return [CameraConfig("base_camera", pose, width, height, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Resample the cube size. _load_scene runs during reconfiguration, after the
        # episode seed has been set, so this stays reproducible from the seed.
        if self.cube_half_size_range is not None:
            low, high = self.cube_half_size_range
            self.cube_half_size = float(self._batched_episode_rng[0].uniform(low, high))

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        if self.render_goal_site:
            self.goal_site = actors.build_sphere(
                self.scene,
                radius=self.goal_thresh,
                color=[0, 1, 0, 1],
                name="goal_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self._hidden_objects.append(self.goal_site)
        else:
            # Build the goal marker with no visual and no collision shape. It still
            # carries the goal pose for evaluate(), the reward terms, the motion
            # planning solutions and the simulation state, but is never rendered.
            # This is preferred over hide_visual(), which on the GPU backend hides an
            # actor by translating it far away and would corrupt the goal position.
            builder = self.scene.create_actor_builder()
            builder.initial_pose = sapien.Pose()
            self.goal_site = builder.build_kinematic(name="goal_site")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            if self.fixed_goal_pos is not None:
                goal_xyz = torch.tensor(
                    self.fixed_goal_pos, dtype=torch.float32
                ).repeat(b, 1)
            else:
                goal_xyz = torch.zeros((b, 3))
                goal_xyz[:, :2] = (
                    torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                    - self.cube_spawn_half_size
                )
                goal_xyz[:, 0] += self.cube_spawn_center[0]
                goal_xyz[:, 1] += self.cube_spawn_center[1]
                goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


PickCubeEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda")


@register_env("PickCubeSO100-v1", max_episode_steps=50)
class PickCubeSO100Env(PickCubeEnv):
    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PickCubeSO100-v1_rt.mp4"

    # The ReCAST cube/goal customization is tuned for the Panda; keep upstream
    # behaviour for the other robots, whose grippers and workspaces differ.
    cube_half_size_range = None
    fixed_goal_pos = None
    render_goal_site = True
    sensor_camera_resolution = (128, 128)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="so100", **kwargs)


PickCubeSO100Env.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="SO100")


@register_env("PickCubeWidowXAI-v1", max_episode_steps=50)
class PickCubeWidowXAIEnv(PickCubeEnv):
    _sample_video_link = "https://github.com/mani-skill/ManiSkill/raw/main/figures/environment_demos/PickCubeWidowXAI-v1_rt.mp4"

    # See PickCubeSO100Env: upstream behaviour is kept for non-Panda robots.
    cube_half_size_range = None
    fixed_goal_pos = None
    render_goal_site = True
    sensor_camera_resolution = (128, 128)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="widowxai", **kwargs)


PickCubeWidowXAIEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="WidowXAI")
