import os
import time
from contextlib import contextmanager
from typing import Literal

import psutil
import torch

import subprocess as sp
def flatten_dict_keys(d: dict, prefix=""):
    """Flatten a dict by expanding its keys recursively."""
    out = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict_keys(v, prefix + k + "/"))
        else:
            out[prefix + k] = v
    return out
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

    def update_csv(self, csv_path: str, data: dict):
        """Update a csv file with the given data (a dict representing a unique identifier of the result row)
        and stats. If the file does not exist, it will be created. The update will replace an existing row
        if the given data matches the data in the row. If there are multiple matches, only the first match
        will be replaced and the rest are deleted"""
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
        for k in data:
            if k not in df:
                df[k] = None

            mask = df[k].isna() if data[k] is None else df[k] == data[k]
            if cond is None:
                cond = mask
            else:
                cond = cond & mask
        data_dict = {**data, **stats_flat}
        if not cond.any():
            df = pd.concat([df, pd.DataFrame(data_dict, index=[len(df)])])
        else:
            # replace the first instance
            df.loc[df.loc[cond].index[0]] = data_dict
            df.drop(df.loc[cond].index[1:], inplace=True)
            # delete other instances
        df.to_csv(csv_path, index=False)

    @contextmanager
    def profile(self, name: str, total_steps: int, num_envs: int):
        print(f"start recording {name} metrics")
        process = psutil.Process(os.getpid())
        cpu_mem_use = process.memory_info().rss
        gpu_mem_use = torch.cuda.mem_get_info()
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
            cpu_mem_use=cpu_mem_use,
            gpu_mem_use=gpu_mem_use[1] - gpu_mem_use[0],
        )
        torch.cuda.synchronize()

    def log_stats(self, name: str):
        stats = self.stats[name]
        self.log(
            f"{name}: {stats['fps']:0.3f} steps/s, {stats['psps']:0.3f} parallel steps/s, {stats['total_steps']} steps in {stats['dt']:0.3f}s"
        )
        self.log(
            f"{' ' * 4}CPU mem: {stats['cpu_mem_use'] / (1024**2):0.3f} MB, GPU mem: {stats['gpu_mem_use'] / (1024**2):0.3f} MB"
        )
