import os
from collections import OrderedDict
from typing import Callable, List, Type

import gym
import numpy as np
from tqdm import tqdm

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import extract_scalars_from_info, merge_dicts
from mani_skill2.utils.io_utils import dump_json, write_txt

from .solution import BasePolicy


class Evaluator:
    env: gym.Env
    policy: BasePolicy

    MAX_EPISODE_STEPS = 1000

    def __init__(self, env_id: str, output_dir: str):
        self.env_id = env_id
        if os.path.exists(output_dir):
            print(f"{output_dir} exists.")
        else:
            os.makedirs(output_dir)
        self.output_dir = output_dir
        self.result = OrderedDict()
        self.merged_metrics = OrderedDict()

    def setup(self, policy_cls: Type[BasePolicy]):
        obs_mode = policy_cls.get_obs_mode(self.env_id)
        control_mode = policy_cls.get_control_mode(self.env_id)
        self.env: BaseEnv = gym.make(
            self.env_id, obs_mode=obs_mode, control_mode=control_mode
        )
        self.policy = policy_cls(
            self.env_id, self.env.observation_space, self.env.action_space
        )

    def evaluate_episode(self, env_kwargs, render_mode=None):
        """Evaluate a single episode."""
        env = self.env
        policy = self.policy

        obs = env.reset(**env_kwargs)
        policy.reset(obs)

        # NOTE(jigu): Use for-loop rather than while-loop
        # in case time limit is not correctly set.
        for _ in range(self.MAX_EPISODE_STEPS):
            action = policy.act(obs)
            # NOTE(jigu): render after action in case action is needed to visualize
            if render_mode is not None:
                env.render(mode=render_mode)
            obs, reward, done, info = env.step(action)
            if done:
                if render_mode is not None:
                    env.render(mode=render_mode)
                assert "success" in info, sorted(info.keys())
                metrics = extract_scalars_from_info(info, "TimeLimit.truncated")
                return metrics

    def evaluate_episodes(self, episode_cfgs: List[dict], callback: Callable = None):
        """
        Evaluate each episode using the config given in each element of episode_cfgs

        Optionally provide a callback that accepts arguments n (episodes completed) and metrics (the results of the latest evaluated episode)
        """
        for i, episode_cfg in enumerate(tqdm(episode_cfgs)):
            episode_id = episode_cfg["episode_id"]
            metrics = self.evaluate_episode(episode_cfg.get("env_kwargs", {}))
            if metrics is None:
                raise RuntimeError(
                    "Episode {}: check whether time limit is set".format(episode_id)
                )
            if episode_id in self.result:
                raise RuntimeError("Episode id {} is not unique.".format(episode_id))
            self.result[episode_id] = metrics
            # TODO(jigu): update progress
            if callback is not None:
                callback(i + 1, metrics)

    def generate_episode_configs(self, num_episodes: int):
        """Generate (dummy) episode configs."""
        return [dict(episode_id=i) for i in range(num_episodes)]

    def merge_result(self):
        merged_result = merge_dicts(self.result.values())
        merged_metrics = {k: np.mean(v) for k, v in merged_result.items()}
        return merged_metrics

    def submit(self):
        # Export per-episode results
        json_path = os.path.join(self.output_dir, "episode_results.json")
        dump_json(json_path, self.result)
        print("The per-episode evaluation result is saved to {}.".format(json_path))

        # Export average result
        json_path = os.path.join(self.output_dir, "average_metrics.json")
        merged_metrics = self.merge_result()
        self.merged_metrics = merged_metrics
        dump_json(json_path, merged_metrics)
        print("The averaged evaluation result is saved to {}.".format(json_path))

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

    def error(self, *args):
        write_txt(os.path.join(self.output_dir, "error.log"), args)
