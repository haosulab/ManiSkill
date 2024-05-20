import numpy as np
import torch

from mani_skill.utils.registration import register_env
from mani_skill.envs.tasks.tabletop import *
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs import Pose
from mani_skill.sensors.camera import CameraConfig
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