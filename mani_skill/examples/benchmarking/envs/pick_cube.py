import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

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

    def compute_dense_reward(self, obs: torch.Any, action: torch.Tensor, info: torch.Dict):
        return info["success"]
