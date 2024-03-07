import time
from contextlib import contextmanager
from typing import Literal

import torch

from mani_skill.utils.common import flatten_dict_keys


class Profiler:
    """
    A simple class to help profile/benchmark simulator code
    """

    def __init__(
        self, output_format: Literal["stdout", "json"], synchronize_torch: bool = True
    ) -> None:
        self.output_format = output_format
        self.synchronize_torch = synchronize_torch
        self.stats = dict()

    def log(self, msg):
        """log a message to stdout"""
        if self.output_format == "stdout":
            print(msg)

    def update_csv(self, csv_path: str, ids: dict):
        import pandas as pd
        import os

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()
        stats_flat = flatten_dict_keys(self.stats)
        cond = None
        for k in stats_flat:
            if k not in df:
                df[k] = None
        for k in ids:
            if k not in df:
                df[k] = None
            if cond is None:
                cond = df[k] == ids[k]
            else:
                cond = cond & (df[k] == ids[k])
        data_dict = {**ids, **stats_flat}
        if not cond.any():
            df.loc[len(df)] = data_dict
        else:
            df.loc[df.loc[cond].index[0]] = data_dict
        df.to_csv(csv_path, index=False)

    @contextmanager
    def profile(self, name: str, total_steps: int, num_envs: int):
        print(f"start recording {name} metrics")
        torch.cuda.synchronize()
        stime = time.time()
        yield
        dt = time.time() - stime
        # dt: delta time (s)
        # fps: frames per second
        # psps: parallel steps per second (number of env.step calls per second)
        self.stats[name] = dict(
            dt=dt,
            fps=total_steps * num_envs / dt,
            psps=total_steps / dt,
            total_steps=total_steps,
        )
        torch.cuda.synchronize()

    def log_stats(self, name: str):
        stats = self.stats[name]
        self.log(
            f"{name}: {stats['fps']:0.3f} steps/s, {stats['psps']:0.3f} parallel steps/s, {stats['total_steps']} steps in {stats['dt']:0.3f}s"
        )
