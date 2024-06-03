from typing import Dict
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig
# TODO (stao): Align this with the isaac lift cube env or the other way around and test RL
@register_env("PickCubeBenchmark-v1")
class PickCubeBenchmarkEnv(PickCubeEnv):
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100,
            control_freq=50,
            scene_cfg=SceneConfig(
                bounce_threshold=0.01,
                solver_position_iterations=8, solver_velocity_iterations=0
            ),
        )

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        return {
            "success": is_obj_placed
        }
    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: torch.Any, action: torch.Tensor, info: torch.Dict):
        return info["success"]
