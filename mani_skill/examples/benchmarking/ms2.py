from typing import Annotated
import gymnasium as gym
import mani_skill2.envs
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from functools import partial
from profiling import Profiler
from mani_skill2.vector import VecEnv, make
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v0"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 16
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "state"

def env_factory(env_id, idx, env_kwargs):
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        return env
    return thunk

def main(args: Args):
    profiler = Profiler(output_format="stdout")
    env_kwargs = dict(obs_mode=args.obs_mode)

    # create a vector env parallelized across CPUs with the given timelimit and auto-reset
    vector_env_cls = partial(AsyncVectorEnv, context="spawn")
    if args.num_envs == 1:
        vector_env_cls = SyncVectorEnv
    env: VectorEnv = vector_env_cls(
        [
            env_factory(
                args.env_id,
                idx,
                env_kwargs=env_kwargs,
            )
            for idx in range(args.num_envs)
        ]
    )
    N = 1000
    env.reset()
    with profiler.profile("env.step", total_steps=N, num_envs=args.num_envs):
        for i in range(N):
            actions = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(actions)
    profiler.log_stats("env.step")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
