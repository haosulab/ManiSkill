import numpy as np
import torch

from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop import *
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig
from mani_skill.utils import sapien_utils

#NOTE: Smaller camera objects render faster

@register_env("PushCube-RandomCameraPose", max_episode_steps=50)
class PushCubeEnvWithRandomCamPose(push_cube.PushCubeEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        # randomize
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        RESOLUTION = (128, 128)
        #RESOLUTION = (256, 256)
        return [CameraConfig("base_camera", pose=pose, width=RESOLUTION[0], height=RESOLUTION[1], fov=np.pi / 2, near=0.01, far=100)]
    
@register_env("PushCube-Pcd", max_episode_steps=50)
class PushCubeEnvWithPointcloud(push_cube.PushCubeEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        RESOLUTION = (128, 128)
        #RESOLUTION = (256, 256)
        cam_cfg = CameraConfig("base_camera", pose=pose, width=RESOLUTION[0], height=RESOLUTION[1], fov=np.pi / 2, near=0.01, far=100)
        return [cam_cfg]
    
@register_env("PullCube-RandomCameraPose", max_episode_steps=50)
class PullCubeEnvWithRandomCamPose(pull_cube.PullCubeEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        # randomize
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        return [CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]
    
@register_env("PickCube-RandomCameraPose", max_episode_steps=50)
class PickCubeEnvWithRandomCamPose(pick_cube.PickCubeEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        # randomize
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        return [CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]
    
@register_env("StackCube-RandomCameraPose", max_episode_steps=50)
class StackCubeEnvWithRandomCamPose(stack_cube.StackCubeEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        # randomize
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        return [CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]
    
@register_env("PegInsertionSide-RandomCameraPose", max_episode_steps=50)
class PegInsertionSideEnvWithRandomCamPose(peg_insertion_side.PegInsertionSideEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        # randomize
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        return [CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]
    
@register_env("AssemblingKits-RandomCameraPose", max_episode_steps=50)
class AssemblingKitsEnvWithRandomCamPose(assembling_kits.AssemblingKitsEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        # randomize
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        return [CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]
    
@register_env("PlugCharger-RandomCameraPose", max_episode_steps=50)
class PlugChargerEnvWithRandomCamPose(plug_charger.PlugChargerEnv):
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)
        # randomize
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )
        return [CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2, near=0.01, far=100)]


import sapien
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.scene import ManiSkillScene

from mani_skill.utils.building.actor_builder import ActorBuilder


def _build_by_type(builder: ActorBuilder, name, body_type):
    if body_type == "dynamic":
        actor = builder.build(name=name)
    elif body_type == "static":
        actor = builder.build_static(name=name)
    elif body_type == "kinematic":
        actor = builder.build_kinematic(name=name)
    else:
        raise ValueError(f"Unknown body type {body_type}")
    return actor

class CustomBuiltPrimitives:
    @staticmethod
    def build_cube(
        scene: ManiSkillScene,
        half_size: float,
        color,
        name: str,
        body_type: str = "dynamic",
        add_collision: bool = True,
        material: sapien.render.RenderMaterial = None,
        physics_properties: dict = {},

    ):
        builder = scene.create_actor_builder()

        if add_collision:
            if physics_properties:
                builder.add_box_collision(
                    half_size=[half_size] * 3,
                    density=physics_properties["density"]
                )
            else:
                builder.add_box_collision(
                    half_size=[half_size] * 3,
                )
        
        builder.add_box_visual(
            half_size=[half_size] * 3,
            material=sapien.render.RenderMaterial(base_color=color,) if material is None else material,
        )
        
        return _build_by_type(builder, name, body_type)

#TODO
@register_env("PushCube-Randomization", max_episode_steps=50)
class PushCubeEnvWithRandomization(push_cube.PushCubeEnv):
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        print(kwargs)
        self.sim_params = kwargs["sim_params"]
        del kwargs["sim_params"]
        self.resolution = self.sim_params["sensor_configs"]
        self.randomize_lights = self.sim_params["randomize_lights"]
        self.randomize_physics = self.sim_params["randomize_physics"]
        self.randomize_material = self.sim_params["randomize_material"]

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        raise("WIP")

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose = Pose.create(pose)

        # randomize
        pose = pose * Pose.create_from_pq(
            p=torch.rand((self.num_envs, 3)) * 0.05 - 0.025,
            q=randomization.random_quaternions(
                n=self.num_envs, device=self.device, bounds=(-np.pi / 24, np.pi / 24)
            ),
        )

        return [CameraConfig(
            "base_camera", 
            pose=pose, 
            width=self.resolution[1], 
            height=self.resolution[0], 
            fov=np.pi / 2, 
            near=0.01, 
            far=100)]
    
    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # NOTE: Randomization
        if self.randomize_lights:
            light_color = self.sim_params["light_color"]
            self.table_scene.scene.set_ambient_light(color=light_color)

        physics_properties = {}
        if self.randomize_physics:
            physics_properties["mass"] = self.sim_params["mass"] #TODO: Add support
            physics_properties["density"] = self.sim_params["density"]

        color = np.array([12, 42, 160, 255]) / 255
        if self.randomize_material:
            specularity = self.sim_params["specularity"]
            metallicity = self.sim_params["metallicity"]
            ior = self.sim_params["ior"]
            transmission = self.sim_params["transmission"]

            custom_material = sapien.render.RenderMaterial(
                base_color=color, 
                metallic=metallicity, 
                specular=specularity,
                ior=ior,
                transmission=transmission
            )

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        self.obj = CustomBuiltPrimitives.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=color,
            name="cube",
            body_type="dynamic",
            material=custom_material,
            physics_properties=physics_properties
        )


        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        # self._hidden_objects.append(self.goal_region)