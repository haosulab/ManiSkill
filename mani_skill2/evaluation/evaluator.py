from collections import OrderedDict
from typing import Callable, List, Type

import gymnasium as gym
import numpy as np

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import extract_scalars_from_info, merge_dicts

from .solution import BasePolicy


class BaseEvaluator:
    env: gym.Env
    policy: BasePolicy

    MAX_EPISODE_STEPS = 1000

    def setup(
        self,
        env_id: str,
        policy_cls: Type[BasePolicy],
        render_mode="cameras",
        env_kwargs=None,
    ):
        """Setup environment and policy."""
        self.env_id = env_id
        self.env_kwargs = {} if env_kwargs is None else env_kwargs

        obs_mode = policy_cls.get_obs_mode(env_id)
        control_mode = policy_cls.get_control_mode(env_id)

        self.env: BaseEnv = gym.make(
            self.env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
            **self.env_kwargs
        )
        self.policy = policy_cls(
            self.env_id, self.env.observation_space, self.env.action_space
        )
        self.result = OrderedDict()

    def evaluate_episode(self, reset_kwargs, render=False):
        """Evaluate a single episode."""
        env = self.env
        policy = self.policy

        obs, _ = env.reset(**reset_kwargs)
        policy.reset(obs)
        # NOTE(jigu): Use for-loop rather than while-loop
        # in case time limit is not correctly set.
        for _ in range(self.MAX_EPISODE_STEPS):
            action = policy.act(obs)
            # NOTE(jigu): render after action in case action is needed to visualize
            if render:
                env.render()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if render:
                    env.render()
                assert "success" in info, sorted(info.keys())
                metrics = extract_scalars_from_info(info, "TimeLimit.truncated")
                return metrics

    def evaluate_episodes(self, episode_cfgs: List[dict], callback: Callable = None):
        """Evaluate episodes according to configurations.

        Args:
            episode_cfgs (List[dict]): a list of episode configurations.
                The configuration should contain "reset_kwargs".
            callback (Callable, optional): callback function to report progress.
                It accepts two arguments:
                    int: the number of completed episodes
                    dict: the results of the latest evaluated episode
        """
        for i, episode_cfg in enumerate(episode_cfgs):
            episode_id = episode_cfg.get("episode_id", i)
            reset_kwargs = episode_cfg.get("reset_kwargs", {})
            metrics = self.evaluate_episode(reset_kwargs)
            if metrics is None:
                raise RuntimeError(
                    "Episode {}: check whether time limit is set".format(episode_id)
                )
            if episode_id in self.result:
                raise RuntimeError("Episode id {} is not unique.".format(episode_id))
            self.result[episode_id] = metrics

            if callback is not None:
                callback(i + 1, metrics)

    def close(self):
        self.env.close()

    def generate_dummy_config(self, env_id, num_episodes: int):
        """Generate dummy configuration."""
        env_info = dict(env_id=env_id)
        episodes = [dict(episode_id=i) for i in range(num_episodes)]
        return dict(env_info=env_info, episodes=episodes)

    def merge_result(self):
        merged_result = merge_dicts(self.result.values())
        merged_metrics = {k: np.mean(v) for k, v in merged_result.items()}
        return merged_metrics

    def export_to_csv(self, path):
        """Average results and export to a csv file."""
        import csv

        import tabulate

        merged_metrics = self.merge_result()
        headers = ["env_id"] + list(merged_metrics.keys())
        data = [[self.env_id] + list(merged_metrics.values())]
        print(tabulate(data, headers=headers, tablefmt="psql", floatfmt=".4f"))

        with open(path, "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)
            csv_writer.writerows(data)
        print("The evaluation result is saved to {}.".format(path))

    def submit(self):
        raise NotImplementedError

    def error(self, *args, **kwargs):
        raise NotImplementedError
