import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mani_skill.envs.tasks.digital_twins.base_real_env import BaseRealEnv
from mani_skill.utils import common, gym_utils, io_utils
from mani_skill.utils.registration import register_env


@register_env("RealPushCube-v1", max_episode_steps=50)
class RealPushCubeEnv(BaseRealEnv):
    sim_env_id = "GrabCube-v1"
    """simulation reference environment"""

    def __init__(
        self,
        yaml_path,
        keyframe_id,
        control_mode,
        control_timing=True,
        image_trans=lambda x: x,
    ):
        super().__init__(
            yaml_path,
            keyframe_id,
            control_mode,
            control_timing=control_timing,
            image_trans=image_trans,
        )

    # NOTE: In emulating maniskill environments, batching obs with batchsize=1 is required
    # agent qpos is already included in state observations, from BaseRealEnv
    def _get_obs_extra(self):
        obs = super()._get_obs_extra()
        qpos = common.batch(self.robot_qpos.clone())
        target_qpos = self.controller._target_qpos.clone()
        obs.update(
            qpos=qpos,
            target_qpos=target_qpos,  # already a batched torch tensor: (1, #joints)
        )
        return obs
