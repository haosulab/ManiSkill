import gymnasium as gym

from .franka import FrankaBenchmarkEnv
gym.register(
    id="Genesis-Franka-Benchmark-v0",
    entry_point="envs.genesis.franka:FrankaBenchmarkEnv",
    disable_env_checker=True
)
