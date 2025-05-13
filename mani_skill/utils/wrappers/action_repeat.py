import torch
import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import Array


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env: BaseEnv, repeat: int):
        """
        Environment wrapper that repeats the action for a number of steps.
        This wrapper will perform the same action at most repeat times, if the environment is done before repeating the action repeat times, then we only return valid data (up to the done=True).

        Args:
            env (BaseEnv): The base environment to wrap.
            repeat (int): The number of times to repeat the action, repeat=1 means no action repeat (we use perform 1 action per step), repeat=2 means the action is repeated twice, so the environment will step twice with the same action.
        """
        super().__init__(env)
        self.repeat = repeat

    @property
    def num_envs(self):
        return self.base_env.num_envs

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def step(self, action):
        final_obs, final_rew, final_terminations, final_truncations, infos = (
            super().step(action)
        )

        is_obs_dict = isinstance(final_obs, dict)

        dones = torch.logical_or(final_terminations, final_truncations)
        not_dones = ~dones

        if not_dones.any():
            for _ in range(self.repeat - 1):
                new_obs, new_rew, new_terminations, new_truncations, new_infos = (
                    super().step(action)
                )

                if is_obs_dict:
                    self._update_dict_values(
                        from_dict=new_obs, to_dict=final_obs, not_dones=not_dones
                    )
                else:
                    final_obs[not_dones] = new_obs[not_dones]

                final_rew[not_dones] += new_rew[not_dones]
                final_terminations[not_dones] = torch.logical_or(
                    final_terminations, new_terminations
                )[not_dones]
                final_truncations[not_dones] = torch.logical_or(
                    final_truncations, new_truncations
                )[not_dones]
                self._update_dict_values(
                    from_dict=new_infos, to_dict=infos, not_dones=not_dones
                )

                dones = torch.logical_or(final_terminations, final_truncations)
                not_dones = ~dones

                if dones.all():
                    break

        return final_obs, final_rew, final_terminations, final_truncations, infos

    def _update_dict_values(self, from_dict: dict, to_dict: dict, not_dones: Array):
        """
        Recursively updates the values of a dictionary with the values from another dictionary but only for the envs that are not done.
        This allows us to update the observation and info dictionaries with new values only for the environments that are not done.
        If a sub-env becomes done, its future step data will be discarded since not_dones will be false for this sub-environment.
        Therefore the final observation/info will come from the true last step of the sub-env.
        """
        for k, v in from_dict.items():
            if isinstance(v, dict):
                self._update_dict_values(
                    from_dict=v, to_dict=to_dict[k], not_dones=not_dones
                )
            elif isinstance(v, Array):
                to_dict[k][not_dones] = v[not_dones]
            else:
                to_dict[k] = v
