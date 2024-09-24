import gymnasium as gym

from .franka_only import FrankaOnlyEnvCfg
from .cartpole_visual import CartpoleRGBCameraBenchmarkEnvCfg
from .cartpole_state import CartpoleEnvCfg
from .franka_cabinet_state import FrankaCabinetEnvCfg
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
    id="Isaac-Franka-Cabinet-Direct-Benchmark-v0",
    entry_point="envs.isaaclab.franka_cabinet_state:FrankaCabinetBenchmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaCabinetEnvCfg,
    },
)

gym.register(
    id="Isaac-Franka-Only-Direct-Benchmark-v0",
    entry_point="envs.isaaclab.franka_only:FrankaOnlyBenchmarkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaOnlyEnvCfg,
    },
)
