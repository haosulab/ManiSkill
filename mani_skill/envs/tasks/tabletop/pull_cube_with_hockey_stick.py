"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, Union

import numpy as np
import torch
import sapien
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array





# _stick_length = 2
# _stick_end_length = 1
# _stick_thickness = 1e-2

# def _build_hockey_stick(
#     scene: ManiSkillScene, 
#     stick_length: float, 
#     end_of_stick_length: float,
# ):
#     builder = scene.create_actor_builder()

#     # stick
#     material = sapien.render.RenderMaterial(
#         base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
#     )

#     builder.add_box_collision()



#     # end of stick (another stick)

from mani_skill.utils.building import articulations







@register_env("PullCubeWithHockeyStick-v1", max_episode_steps=50)
class PullCubeWithHockeyStickEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02

    # same as pick_cube, stack_cube and push_cube
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    # same as pick_cube, stack_cube and push_cube
    @property 
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]


    # same as pick_cube, stack_cube and push_cube
    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )
        model_id = "4000" # from info_bucket_train.json
        builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{model_id}"
            )
        self.bucket = builder.build(name="bottle")

        

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # initial setup for handling multiple environments
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # set the cube's initial position
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # set the bottle's initial position
            print("self.bottle: ", self.bottle)
            print("self.bottle attributes: ", dir(self.bottle))
            target_region_xyz = xyz + torch.tensor([0.1 + self.goal_radius, 0, 0])
            target_region_xyz[..., 2] = 1e-3
            self.bottle.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        return {
            "success": True,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            pass
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        return 5

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        return 5