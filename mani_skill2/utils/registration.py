import re

import gym

REGISTERED_ENV_IDS = set()
env_id_re = re.compile(r"^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$")


def register(name, entry_point, max_episode_steps, kwargs):
    """A wrapper of gym.register."""
    if name in REGISTERED_ENV_IDS:
        gym.logger.warn(f"{name} is registered in gym already!")
    else:
        REGISTERED_ENV_IDS.add(name)
        gym.register(
            name,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            kwargs=kwargs,
        )


def register_gym_env(name: str, max_episode_steps=None, **kwargs):
    """A decorator to register ManiSkill environments in gym.

    Args:
        name (str): a unique id to register in gym.

    Notes:
        ManiSkill envs should maintain `max_episode_steps` by itself, rather than gym TimeLimit wrapper.
        `gym.EnvSpec` uses kwargs instead of **kwargs!
    """

    def _register_gym_env(cls):
        entry_point = "{}:{}".format(cls.__module__, cls.__name__)

        register(
            name,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            kwargs=kwargs,
        )

        # Register different observation modes for simplicity
        if hasattr(cls, "SUPPORTED_OBS_MODES"):
            match = env_id_re.search(name)
            env_name = match.group(1)
            env_version = match.group(2)
            for obs_mode in cls.SUPPORTED_OBS_MODES:
                env_kwargs = {"obs_mode": obs_mode}
                env_kwargs.update(kwargs)
                env_id = f"{env_name}-{obs_mode}-v{env_version}"
                register(
                    env_id,
                    entry_point=entry_point,
                    max_episode_steps=max_episode_steps,
                    kwargs=kwargs,
                )

        return cls

    return _register_gym_env
