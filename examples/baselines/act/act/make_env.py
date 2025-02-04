from typing import Optional
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers import RecordEpisode, CPUGymWrapper


def make_eval_envs(env_id, num_envs: int, sim_backend: str, env_kwargs: dict, other_kwargs: dict, video_dir: Optional[str] = None, wrappers: list[gym.Wrapper] = []):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":
        def cpu_make_env(env_id, seed, video_dir=None, env_kwargs = dict(), other_kwargs = dict()):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, info_on_video=True, source_type="act", source_desc="act evaluation rollout")
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk
        vector_cls = gym.vector.SyncVectorEnv if num_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
        env = vector_cls([cpu_make_env(env_id, seed, video_dir if seed == 0 else None, env_kwargs, other_kwargs) for seed in range(num_envs)])
    else:
        env = gym.make(env_id, num_envs=num_envs, sim_backend=sim_backend, reconfiguration_freq=1, **env_kwargs)
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        if video_dir:
            env = RecordEpisode(env, output_dir=video_dir, save_trajectory=False, save_video=True, source_type="act", source_desc="act evaluation rollout", max_steps_per_video=max_episode_steps)
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env
