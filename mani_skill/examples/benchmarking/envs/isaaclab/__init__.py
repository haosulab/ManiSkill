import gymnasium as gym

from .franka import FrankaEnvCfg
from .cartpole_visual import CartpoleRGBCameraBenchmarkEnvCfg
from .cartpole_state import CartpoleEnvCfg
gym.register(
    id="Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0",
    entry_point="envs.isaaclab.cartpole_visual:CartpoleCameraBenchmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleRGBCameraBenchmarkEnvCfg,
    },
)
gym.register(
    id="Isaac-Cartpole-Direct-Benchmark-v0",
    entry_point="envs.isaaclab.cartpole_state:CartpoleBenchmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
    },
)

gym.register(
    id="Isaac-Franka-Direct-Benchmark-v0",
    entry_point="envs.isaaclab.franka:FrankaBenchmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaEnvCfg,
    },
)
