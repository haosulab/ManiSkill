import gymnasium as gym

from mani_skill2.utils.common import flatten_dict_keys, flatten_dict_space_keys


class BaseGymObservationWrapper(gym.ObservationWrapper):
    """ManiSkill2 uses a custom registration function that uses observation wrappers to change observation mode based on env kwargs.
    By default gymnasium does not expect custom registration and so creating an with gymnasium may sometimes raise an error as it tries to set the spec of an env
    which is possible if the registered env is not wrapped.
    """

    @property
    def spec(self):
        return self.unwrapped.spec

    @spec.setter
    def spec(self, spec):
        self.unwrapped.spec = spec


# TODO (stao): is this still needed?
class FlattenObservationWrapper(BaseGymObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.observation_space = flatten_dict_space_keys(self.observation_space)

    def observation(self, observation):
        return flatten_dict_keys(observation)
