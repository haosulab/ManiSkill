import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mani_skill.envs.tasks.digital_twins.utils.real_env import BaseRealEnv
from mani_skill.utils import common, gym_utils, io_utils
from mani_skill.utils.registration import register_env


@register_env("RealGrabCube-v1", max_episode_steps=75)
class RealGrabCubeEnv(BaseRealEnv):
    sim_env_id = "GrabCube-v1"
    """simulation reference environment"""

    def __init__(self, yaml_path, keyframe_id, control_mode, image_trans=lambda x: x):
        super().__init__(yaml_path, keyframe_id, control_mode, image_trans=image_trans)

    # NOTE: In emulating maniskill environments, batching obs is required
    # agent qpos is already included in state observations, from BaseRealEnv
    def _get_obs_extra(self):
        obs = super()._get_obs_extra()
        obs.update(
            # to_rest_dist=self.rest_qpos[:-1] - self.agent.robot.qpos[..., :-1],
            to_rest_dist=common.batch(
                self.robot_keyframe_qpos[:-1].clone() - self.robot_qpos[:-1]
            ),
            rest_qpos=common.batch(
                self.robot_keyframe_qpos[:-1].clone()
            ),  # already a torch tensor, need to batch: (#joints) -> (1, #joints)
            target_qpos=self.controller._target_qpos.clone(),  # already a batched torch tensor
        )
        return obs
