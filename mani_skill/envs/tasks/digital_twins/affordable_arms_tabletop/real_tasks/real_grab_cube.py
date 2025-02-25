import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mani_skill.envs.tasks.digital_twins.base_real_env import BaseRealEnv
from mani_skill.envs.tasks.digital_twins.real_agents.koch import MS3RealKoch
from mani_skill.utils import common
from mani_skill.utils.registration import register_env


@register_env("RealGrabCube-v1", max_episode_steps=75)
class RealGrabCubeEnv(BaseRealEnv):
    real_agent_cls = MS3RealKoch

    def __init__(
        self,
        agent=None,
        keyframe_id=None,
        control_freq=None,
        control_mode=None,
        control_timing=True,
        **kwargs
    ):
        if agent is None:
            agent = self.real_agent_cls(
                control_freq=control_freq, control_mode=control_mode, **kwargs
            )
        # we default to koch's 'evelated_turn' keyframe for this task
        super().__init__(
            agent, keyframe_id="elevated_turn", control_timing=control_timing
        )

    # NOTE: In emulating maniskill environments, batching obs with batchsize=1 is required
    # agent qpos is already included in state observations, from BaseRealEnv
    def _get_obs_extra(self):
        obs = super()._get_obs_extra()
        qpos = common.batch(self.agent.qpos.clone())
        target_qpos = self.agent.controller._target_qpos.clone()
        is_grasped = ((qpos[..., -1] - target_qpos[..., -1]) >= 0.02).float() * (
            target_qpos[..., -1] < 0.24
        )
        obs.update(
            qpos=qpos,
            to_rest_dist=common.batch(
                self.robot_keyframe_qpos[:-1].clone() - qpos[0, :-1]
            ),
            rest_qpos=common.batch(
                self.robot_keyframe_qpos[:-1].clone()
            ),  # already a torch tensor, need to batch: (#joints) -> (1, #joints)
            target_qpos=target_qpos,  # already a batched torch tensor: (1, #joints)
            is_grasped=is_grasped.view(1, 1),
        )
        return obs
