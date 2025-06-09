import gc
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import psutil
import pynvml
import torch

import mani_skill.envs


def get_current_process_gpu_memory(gpu_handle, current_pid):
    # Get all processes running on the GPU
    if gpu_handle is not None:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)
        # Iterate through the processes to find the current process
        for process in processes:
            if process.pid == current_pid:
                memory_usage = process.usedGpuMemory
                return memory_usage


def main():
    gpu_handle = None
    try:
        pynvml.nvmlInit()
        # Get handle for the first GPU (index 0)
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError_LibraryNotFound as e:
        print(
            f"NVML could not be initialized for pynvml: {e}. Not tracking GPU memory anymore."
        )

    current_pid = os.getpid()
    process = psutil.Process()
    env = gym.make("PickCube-v1", num_envs=256, obs_mode="state", reward_mode="sparse")
    base_env = env.unwrapped
    action = (
        torch.from_numpy(base_env.action_space.sample()).to(base_env.device).float()
    )
    env.reset(seed=0)
    start_cpu_mem = process.memory_info().rss
    start_gpu_mem = get_current_process_gpu_memory(gpu_handle, current_pid)
    i = 0

    freq = 50
    n_samples = [None] * 10_000
    cpu_memory = [None] * 10_000
    gpu_memory = [None] * 10_000

    while True:
        # env.reset(seed=i, options=dict(reconfigure=True))
        # for _ in range(10):
        base_env.step(action)
        # env.action_space.sample()
        # env.unwrapped.scene.px.cuda_rigid_body_data  # this is fine, no increase in mem
        # env.unwrapped.scene.px.cuda_rigid_body_data.torch()  # this increases memory on torch 2.6.0
        i += 1

        if i % freq == 0:
            # gc.collect()
            # if i % freq == 0:
            #     cpu_mem_at_10 = process.memory_info().rss
            #     gpu_mem_at_10 = get_current_process_gpu_memory(gpu_handle, current_pid)

            cpu_mem = process.memory_info().rss
            gpu_mem = get_current_process_gpu_memory(gpu_handle, current_pid)
            print(
                f"Iteration {i}\nCPU memory: {cpu_mem / 1024 / 1024:0.02f}MB; {cpu_mem / start_cpu_mem:0.04f}x change;"
            )
            print(
                f"GPU memory: {gpu_mem / 1024 / 1024:0.02f}MB; {gpu_mem / start_gpu_mem:0.04f}x change;"
            )

            n_samples[i // freq] = i
            cpu_memory[i // freq] = cpu_mem
            gpu_memory[i // freq] = gpu_mem

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # CPU Memory Plot
            ax1.plot(n_samples[: i // freq + 1], cpu_memory[: i // freq + 1])
            ax1.set_title("CPU Memory Usage")
            ax1.set_xlabel("Steps")
            ax1.set_ylabel("Memory (GB)")
            ax1.grid(True)

            # GPU Memory Plot
            # ax2.plot(n_samples[:i // freq + 1], gpu_memory[:i // freq + 1])
            # ax2.set_title("GPU Memory Usage")
            # ax2.set_xlabel("Steps")
            # ax2.set_ylabel("Memory (GB)")
            # ax2.grid(True)

            plt.tight_layout()
            plt.savefig(f"memory_usage_nothing.png")
            plt.close()


if __name__ == "__main__":
    main()
