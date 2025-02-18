import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from gymnasium.vector.vector_env import VectorEnvWrapper
from mani_skill.utils.structs.types import Array
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


class ActionRepeatWrapper(VectorEnvWrapper):
    def __init__(self, env: ManiSkillVectorEnv, repeat: int):
        """Vectorized environment wrapper that repeats the action for a number of steps.
        Args:
            env (ManiSkillVectorEnv): The base environment to wrap.
            repeat (int): The number of times to repeat the action, repeat=1 means no action repeat (we use the action to perform 1 step), repeat=2 means the action is repeated twice, so the environment will step twice with the same action.
        """
        super().__init__(env)
        self.repeat = repeat
        self._auto_reset = (
            env.auto_reset
        )  # The wrapper will handle auto resets based on if the base env had auto_reset enabled
        self.env.auto_reset = False # Disable auto_reset in the base env since we will handle it in this wrapper

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
    ):
        return self.env.reset(seed=seed, options=options)

    def step(
        self, actions: Union[Array, Dict]
    ) -> Tuple[Array, Array, Array, Array, Dict]:
        final_obs, final_rew, final_terminations, final_truncations, infos = (
            self.env.step(actions)
        )

        is_obs_dict = isinstance(final_obs, dict)

        dones = torch.logical_or(final_terminations, final_truncations)
        not_dones = ~dones

        if not dones.all():
            for _ in range(self.repeat - 1):
                new_obs, new_rew, new_terminations, new_truncations, new_infos = self.env.step(actions)

                if is_obs_dict:
                    self._update_dict_values(from_dict=new_obs, to_dict=final_obs, not_dones=not_dones)
                else:
                    final_obs[not_dones] = new_obs[not_dones]
                final_rew[not_dones] += new_rew[not_dones]
                final_terminations[not_dones] = torch.logical_or(final_terminations, new_terminations)[not_dones]
                final_truncations[not_dones] = torch.logical_or(final_truncations, new_truncations)[not_dones]
                self._update_dict_values(from_dict=new_infos, to_dict=infos, not_dones=not_dones)

                dones = torch.logical_or(final_terminations, final_truncations)
                not_dones = ~dones

                if dones.all():
                    break

        if dones.any() and self._auto_reset:
            obs, infos = self.env._do_auto_reset(dones=dones, final_obs=final_obs, infos=infos)
        else:
            obs = final_obs

        return obs, final_rew, final_terminations, final_truncations, infos

    def _update_dict_values(self, from_dict: dict, to_dict: dict, not_dones: Array):
        """
        Recursively updates the values of a dictionary with the values from another dictionary but only for the envs that are not done.
        This allows us to update the observation and info dictionaries with new values only for the environments that are not done.
        If a sub-env becomes done, its future step data will be discarded since not_dones will be false for this sub-environment.  
        Therefore the final observation/info will come from the true last step of the sub-env.
        """
        for k, v in from_dict.items():
            if isinstance(v, dict):
                self._update_dict_values(from_dict=v, to_dict=to_dict[k], not_dones=not_dones)
            elif isinstance(v, Array):
                to_dict[k][not_dones] = v[not_dones]
            else:
                to_dict[k] = v
