from typing import Any, Dict, List, Union, Tuple

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig        
8
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Actor

from sapien.physx import PhysxRigidBodyComponent
from sapien.render import RenderBodyComponent

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("PickCube-v1", max_episode_steps=50)
class PickCubeEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[XArm6Robotiq]
    cube_half_size = 0.02
    goal_thresh = 0.025  
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
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
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        # return [CameraConfig("base_camera", pose, 84, 84, np.pi / 2, 0.01, 100)]

        return [CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, -0.05], q=[0.70710678, 0, 0.70710678, 0]),
                width=84,
                height=84,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["camera_link"],
            )]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict, initial_agent_poses = sapien.Pose(p=[-0.615, 0, 0]), build_separate: bool = False):
        super()._load_agent(options, initial_agent_poses, build_separate)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise, custom_table=True
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
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
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            # tcp_pose=self.agent.tcp.pose.raw_pose,
            # goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.cube.pose.raw_pose,
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

    def staged_rewards(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        place_reward *= is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        static_reward *= info["is_obj_placed"]

        return reaching_reward.mean(), is_grasped.mean(), place_reward.mean(), static_reward.mean()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
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
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


PickCubeEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda")


@register_env("PickCubeSO100-v1", max_episode_steps=50)
class PickCubeSO100Env(PickCubeEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCubeSO100-v1_rt.mp4"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="so100", **kwargs)


PickCubeSO100Env.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="SO100")


@register_env("PickCubeWidowXAI-v1", max_episode_steps=50)
class PickCubeWidowXAIEnv(PickCubeEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCubeWidowXAI-v1_rt.mp4"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="widowxai", **kwargs)


PickCubeWidowXAIEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="WidowXAI")


@register_env("PickCubeDR-v1", max_episode_steps=50)
class PickCubeDR(PickCubeEnv):
    def __init__(self, *args,  robot_uids="xarm6_robotiq", robot_init_qpos_noise=0.02, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, reconfiguration_freq=1, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, initial_agent_poses=sapien.Pose(), build_separate=True)

    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light(np.random.uniform(-1, 1, 3), [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: dict):
        '''
            Custom load_scene where every parallel environment has a different color for the cube.
        '''
        ### Table randomization, handled in TableSceneBuilder
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise, 
            custom_table=True, randomize_colors=True
        )
        self.table_scene.build()

        ### Cube randomization: Build cubes separately for each parallel environment to enable domain randomization        
        self._cubes: List[Actor] = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[self.cube_half_size] * 3)
            builder.add_box_visual(
                half_size=[self.cube_half_size] * 3, 
                material=sapien.render.RenderMaterial(
                    base_color=self._batched_episode_rng[i].uniform(low=0., high=1., size=(3, )).tolist() + [1]
                )
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
            builder.set_scene_idxs([i])
            self._cubes.append(builder.build(name=f"cube_{i}"))
            self.remove_from_state_dict_registry(self._cubes[-1])  # remove individual cube from state dict

        # Merge all cubes into a single Actor object
        self.cube = Actor.merge(self._cubes, name="cube")
        self.add_to_state_dict_registry(self.cube)  # add merged cube to state dict

        ### Agent randomization
        for link in self.agent.robot.links:
            for i, obj in enumerate(link._objs):
                # modify the i-th object which is in parallel environment i
                
                # modifying physical properties e.g. randomizing mass from 0.1 to 1kg
                rigid_body_component: PhysxRigidBodyComponent = obj.entity.find_component_by_type(PhysxRigidBodyComponent)
                if rigid_body_component is not None:
                    # note the use of _batched_episode_rng instead of torch.rand. _batched_episode_rng helps ensure reproducibility in parallel environments.
                    rigid_body_component.mass = self._batched_episode_rng[i].uniform(low=0.1, high=1)
                
                # modifying per collision shape properties such as friction values
                for shape in obj.collision_shapes:
                    shape.physical_material.dynamic_friction = self._batched_episode_rng[i].uniform(low=0.1, high=0.3)
                    shape.physical_material.static_friction = self._batched_episode_rng[i].uniform(low=0.1, high=0.3)
                    shape.physical_material.restitution = self._batched_episode_rng[i].uniform(low=0.1, high=0.3)

                render_body_component: RenderBodyComponent = obj.entity.find_component_by_type(RenderBodyComponent)
                if render_body_component is not None:
                    for render_shape in render_body_component.render_shapes:
                        for part in render_shape.parts:
                            # you can change color, use texture files etc.
                            part.material.set_base_color(self._batched_episode_rng[i].uniform(low=0., high=1., size=(3, )).tolist() + [1])
                            # note that textures must use the sapien.render.RenderTexture2D 
                            # object which allows passing a texture image file path
                            part.material.set_base_color_texture(None)
                            part.material.set_normal_texture(None)
                            part.material.set_emission_texture(None)
                            part.material.set_transmission_texture(None)
                            part.material.set_metallic_texture(None)
                            part.material.set_roughness_texture(None)

        ### Non-randomized objects
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

    def _reconfigure(self, options=dict()):
        """Clean up individual actors created for domain randomization to prevent memory leaks during resets."""
        if hasattr(self, '_cubes'):
            # Remove individual cubes from the scene
            for cube in self._cubes:
                if hasattr(cube, 'entity') and cube.entity is not None:
                    self.scene.remove_actor(cube)
            self._cubes.clear()
        
        # Clean up table scene builder if it exists
        if hasattr(self, 'table_scene'):
            self.table_scene.cleanup()

        super()._reconfigure(options)

PickCubeDR.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="xarm6_robotiq")


class DiscreteInitMixin:
    """Mixin that supports *discrete* environment initialisation on an N×N grid.

    The grid resolution can be configured via the ``grid_dim`` argument that is
    forwarded to the environment constructor.  By default we reproduce the
    previous behaviour with ``grid_dim=10`` (100 unique environments).
    """

    # ---------------------------------------------------------------------
    # Constructor – stores the grid dimension before calling the parent __init__
    # ---------------------------------------------------------------------
    def __init__(self, *args, grid_dim: int = 10, **kwargs):
        self.grid_dim = grid_dim  # number of cells per axis (grid_dim x grid_dim)
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Helper to convert a linear env-index → (x,y) grid coordinates
    # ---------------------------------------------------------------------
    def _index_to_xy(self, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_grid = idx // self.grid_dim
        y_grid = idx %  self.grid_dim
        return x_grid, y_grid

    # ---------------------------------------------------------------------
    # Episode initialisation
    # ---------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Place cube and goal according to the unique *environment index*.

        The caller can supply a key ``global_idx`` in *options* containing a
        1-D list/array/torch.Tensor of length = ``len(env_idx)`` that specifies
        which global (x, y) cell each parallel environment should correspond
        to.  If absent, we fall back to the local ``env_idx`` numbering (0…K-1).
        """
        with torch.device(self.device):
            # Resolve the *effective* indices for this reset -------------------
            if options is not None and "global_idx" in options:
                gidx = options["global_idx"]
                if isinstance(gidx, torch.Tensor):
                    gidx = gidx.to(env_idx.device)
                else:
                    gidx = torch.as_tensor(gidx, device=env_idx.device)
                assert len(gidx) == len(env_idx), "global_idx length mismatch"
                eff_idx = gidx.long()
            else:
                eff_idx = env_idx.long()

            b = len(eff_idx)
            self.table_scene.initialize(env_idx)

            # Buffer for cube positions (B,3)
            xyz = torch.zeros((b, 3), device=env_idx.device)

            # Map env-indices → discrete grid coordinates
            x_grid, y_grid = self._index_to_xy(eff_idx)

            # Equally-spaced centres along one axis (size = grid_dim)
            lin = torch.linspace(
                -self.cube_spawn_half_size,
                 self.cube_spawn_half_size,
                 self.grid_dim,
                 device=env_idx.device,
            )

            # Assign positions and apply centre offset
            xyz[:, 0] = lin[x_grid] + self.cube_spawn_center[0]
            xyz[:, 1] = lin[y_grid] + self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size

            # Random cube orientation (axis-aligned)
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Goal site placed 20 cm above the cube
            goal_xyz = xyz.clone()
            goal_xyz[:, 2] = xyz[:, 2] + 0.2
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))



@register_env("PickCubeDiscreteInit-v1", max_episode_steps=50)
class PickCubeDiscreteInit(DiscreteInitMixin, PickCubeEnv):
    def __init__(self, *args, robot_uids="xarm6_robotiq", grid_dim: int = 10, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0.0, grid_dim=grid_dim, **kwargs)


PickCubeDiscreteInit.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="xarm6_robotiq")


@register_env("PickCubeDRDiscreteInit-v1", max_episode_steps=50)
class PickCubeDRDiscreteInit(DiscreteInitMixin, PickCubeDR):
    def __init__(self, *args, robot_uids="xarm6_robotiq", grid_dim: int = 10, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=0.0, grid_dim=grid_dim, **kwargs)


PickCubeDRDiscreteInit.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="xarm6_robotiq")