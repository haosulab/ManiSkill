import torch
from mani_skill.envs.tasks.control.cartpole import CartpoleBalanceEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

@register_env("CartpoleBalanceBenchmark-v1")
class CartPoleBalanceBenchmarkEnv(CartpoleBalanceEnv):
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=120,
            spacing=20,
            control_freq=60,
            scene_cfg=SceneConfig(
                bounce_threshold=0.5,
                solver_position_iterations=4, solver_velocity_iterations=0
            ),
        )
    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)
