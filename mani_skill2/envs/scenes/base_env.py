import numpy as np
import sapien as sapien
import sapien.physx as physx
from sapien import Pose

from mani_skill2.agents.robots import Panda
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.scene_builder import SceneBuilder
from mani_skill2.utils.scene_builder.ai2thor import (
    ArchitecTHORSceneBuilder,
    ProcTHORSceneBuilder,
    RoboTHORSceneBuilder,
    iTHORSceneBuilder,
)
from mani_skill2.utils.structs.pose import vectorize_pose


class SceneManipulationEnv(BaseEnv):
    agent: Panda
    """
    Args:
        robot_uid: Which robot to place into the scene. Default is "panda"

        fixed_scene: whether to sample a single scene and never reconfigure the scene during episode resets
        Default to True as reconfiguration/reloading scenes is expensive. When true, call env.reset(seed=seed, options=dict(reconfigure=True))

        scene_builder_cls: Scene builder class to build a scene with. Default is the ArchitecTHORSceneBuilder which builds a scene from AI2THOR. 
            Any of the AI2THOR SceneBuilders are supported in this environment

        convex_decomposition: Choice of convex decomposition algorithm to generate collision meshes for objects. Default is `coacd` which uses https://github.com/SarahWeiii/CoACD
    """

    def __init__(
        self,
        *args,
        robot_uid="panda",
        robot_init_qpos_noise=0.02,
        fixed_scene=True,
        scene_builder_cls: SceneBuilder = ArchitecTHORSceneBuilder,
        convex_decomposition="coacd",
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.fixed_scene = fixed_scene
        self.sampled_scene_idx: int = None
        self.scene_builder = scene_builder_cls()
        self.scene_ids = np.arange(0, len(self.scene_builder.scene_configs))
        self.convex_decomposition = convex_decomposition
        super().__init__(*args, robot_uid=robot_uid, **kwargs)

    def reset(self, seed=None, options=None):
        self._set_episode_rng(seed)
        if options is None:
            options = dict(reconfigure=True)
        if not self.fixed_scene:
            options["reconfigure"] = True
        if options["reconfigure"]:
            self.sampled_scene_idx = self._episode_rng.randint(0, len(self.scene_ids))
        return super().reset(seed, options)

    def _load_actors(self):
        self.scene_builder.build(
            self._scene,
            scene_id=self.sampled_scene_idx,
            convex_decomposition=self.convex_decomposition,
        )

    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            # EE at [0.615, 0, 0.17]
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        elif self.robot_uid == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    def _register_sensors(self):
        pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return CameraConfig(
            "base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

    def _register_render_cameras(self):
        if self.robot_uid == "panda":
            pose = look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        else:
            pose = look_at([0.5, 0.5, 1.0], [0.0, 0.0, 0.5])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs
