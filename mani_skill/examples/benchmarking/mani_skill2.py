import gymnasium as gym
import mani_skill2.envs
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from functools import partial
from profiling import Profiler
from mani_skill2.vector import VecEnv, make
# env: VecEnv = make("PickCube-v0", num_envs=4)

if __name__ == "__main__":
    profiler = Profiler(output_format="stdout")
    env_id = "OpenCabinetDrawer-v1"
    def env_factory(env_id, idx, env_kwargs):
        def thunk():
            env = gym.make(env_id, **env_kwargs)
            return env
        return thunk
    # env = gym.make("OpenCabinetDrawer-v1")
    # env.reset()
    context = "forkserver"
    num_envs = 16
    env_kwargs = dict(obs_mode="rgbd")
     # create a vector env parallelized across CPUs with the given timelimit and auto-reset
    vector_env_cls = partial(AsyncVectorEnv, context=context)
    if num_envs == 1:
        vector_env_cls = SyncVectorEnv
    env: VectorEnv = vector_env_cls(
        [
            env_factory(
                env_id,
                idx,
                env_kwargs=env_kwargs,
            )
            for idx in range(num_envs)
        ]
    )
    # env: VecEnv = make(env_id, num_envs=num_envs, **env_kwargs)
    env.reset()
    N = 100
    with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
        for i in range(N):
            actions = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(actions)
    profiler.log_stats("env.step")
