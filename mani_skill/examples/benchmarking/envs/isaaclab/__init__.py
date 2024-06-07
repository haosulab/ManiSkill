import gymnasium as gym
from .cartpole_rgb import CartpoleRGBCameraBenchmarkEnvCfg
from .cartpole_state import CartpoleEnvCfg
gym.register(
    id="Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0",
    entry_point="envs.isaaclab.cartpole_rgb:CartpoleCameraBenchmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleRGBCameraBenchmarkEnvCfg,
    },
)
from .cartpole_rgb import CartpoleRGBCameraBenchmarkEnvCfg
gym.register(
    id="Isaac-Cartpole-Direct-Benchmark-v0",
    entry_point="envs.isaaclab.cartpole_state:CartpoleBenchmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
    },
)
