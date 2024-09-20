import os
import time
from contextlib import contextmanager
from typing import Literal
import numpy as np

import psutil
import torch
import pynvml
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
        # Initialize NVML
        pynvml.nvmlInit()

        # Get handle for the first GPU (index 0)
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Get the PID of the current process
        self.current_pid = os.getpid()

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
        gpu_mem_use = self.get_current_process_gpu_memory()
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
            gpu_mem_use=gpu_mem_use,
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

    def get_current_process_gpu_memory(self):
        # Get all processes running on the GPU
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)

        # Iterate through the processes to find the current process
        for process in processes:
            if process.pid == self.current_pid:
                memory_usage = process.usedGpuMemory
                return memory_usage

def tile_images(images, nrows=1):
    """
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns to fit all the images.
    The images can also be batched (e.g. of shape (B, H, W, C)), but give images must all have the same batch size.

    if nrows is 1, images can be of different sizes. If nrows > 1, they must all be the same size.
    """
    # Sort images in descending order of vertical height
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched : 2 + batched]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1 + batched] for x in columns)

    is_torch = False
    if torch is not None:
        is_torch = isinstance(images[0], torch.Tensor)

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    if is_torch:
        output_image = torch.zeros(output_shape, dtype=images[0].dtype)
    else:
        output_image = np.zeros(output_shape, dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        if is_torch:
            column_image = torch.concatenate(column, dim=0 + batched)
        else:
            column_image = np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        output_image[..., :cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image
