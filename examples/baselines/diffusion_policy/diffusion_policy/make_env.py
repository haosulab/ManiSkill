from typing import Optional
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from diffusion_policy.wrappers import SeqActionWrapper


def make_env(env_id, num_envs: int, sim_backend: str, seed: int, env_kwargs: dict, other_kwargs: dict,video_dir: Optional[str] = None):
    if sim_backend == "cpu":
        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs = dict(), other_kwargs = dict()):
            def thunk():
                env = gym.make(env_id, **env_kwargs)
                env = CPUGymWrapper(env)
                if video_dir:
                    env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True)

                env = gym.wrappers.RecordEpisodeStatistics(env)
                env = gym.wrappers.ClipAction(env)
                env = gym.wrappers.FrameStack(env, other_kwargs['obs_horizon'])
                env = SeqActionWrapper(env)

                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk
        vector_cls = gym.vector.SyncVectorEnv if num_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
        env = vector_cls([cpu_make_env(env_id, seed, video_dir if seed == 0 else None, env_kwargs, other_kwargs) for seed in range(num_envs)])
    else:
        env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend, **env_kwargs)
    return env
