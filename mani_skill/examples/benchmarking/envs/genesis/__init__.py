import gymnasium as gym

from .franka_move import FrankaMoveBenchmarkEnv
from .franka_pick_cube import FrankaPickCubeBenchmarkEnv

gym.register(
    id="Genesis-FrankaMove-Benchmark-v0",
    entry_point="envs.genesis.franka_move:FrankaMoveBenchmarkEnv",
    disable_env_checker=True
)

gym.register(
    id="Genesis-FrankaPickCube-Benchmark-v0",
    entry_point="envs.genesis.franka_pick_cube:FrankaPickCubeBenchmarkEnv",
    disable_env_checker=True
)
