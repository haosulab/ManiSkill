import gymnasium as gym

from .franka import FrankaBenchmarkEnv
from .franka_pick_cube import FrankaPickCubeBenchmarkEnv

gym.register(
    id="Genesis-Franka-Benchmark-v0",
    entry_point="envs.genesis.franka:FrankaBenchmarkEnv",
    disable_env_checker=True
)

gym.register(
    id="Genesis-FrankaPickCube-Benchmark-v0",
    entry_point="envs.genesis.franka_pick_cube:FrankaPickCubeBenchmarkEnv",
    disable_env_checker=True
)
