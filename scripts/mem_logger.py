import numpy as np
import os
import psutil as ps
import time
import torch

class MemLogger(object):
    def __init__(self, device, dump_after, gpu_mem_logs_path, cpu_mem_logs_path, loss_logs_path, lbl):
        self.device = device
        assert self.device == 'cuda', "gpu should be set"

        self.dump_after = dump_after

        self.peak_gpu_mem_logs = []
        self.peak_cpu_mem_logs = []
        self.loss_logs = []

        self.peak_gpu_mem_timestamps = []
        self.peak_cpu_mem_timestamps = []
        self.loss_timestamps = []

        self.peak_gpu_mem_logs_path = gpu_mem_logs_path
        self.peak_cpu_mem_logs_path = cpu_mem_logs_path
        self.loss_logs_path = loss_logs_path
        self.lbl = lbl

        self.bytes_to_mebibytes = 1024 * 1024 # or 1 << 20 or 2**20 (bytes to mebibytes)

    def get_gpu_allocated_memory(self):
        memory = torch.cuda.memory_allocated(self.device) / self.bytes_to_mebibytes
        return memory
    
    def get_cpu_allocated_memory(self):
        memory = ps.Process().memory_info().rss / self.bytes_to_mebibytes
        return memory

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()

        self.peak_gpu_mem_logs.append(torch.cuda.memory_allocated(self.device) / self.bytes_to_mebibytes)
        self.peak_gpu_mem_timestamps.append(time.time())

        self.peak_cpu_mem_logs.append(ps.Process().memory_info().rss / self.bytes_to_mebibytes)
        self.peak_cpu_mem_timestamps.append(time.time())

    def __exit__(self, exception_type, exception_value, exception_traceback):
        bytes_to_mebibytes = 1024 * 1024 # or 1 << 20 or 2**20  (bytes to mebibytes)

        self.peak_gpu_mem_logs.append(torch.cuda.memory_allocated(self.device) / bytes_to_mebibytes)
        self.peak_gpu_mem_timestamps.append(time.time())

        self.peak_cpu_mem_logs.append(ps.Process().memory_info().rss / bytes_to_mebibytes)
        self.peak_cpu_mem_timestamps.append(time.time())

    def log_loss(self, loss):
        """Log RecMSE at each epoch"""
        self.loss_logs.append(loss)
        self.loss_timestamps.append(time.time())

    def dump(self):
        if len(self.loss_logs) > self.dump_after - 1:
            print("Dumping measures")
            self._log_gpu_mem()
            self._log_cpu_mem()
            self._log_loss()

    def close(self):
        if len(self.peak_gpu_mem_logs) > 0:
            print("Dumping gpu leftover measures")
            self._log_gpu_mem()

        if len(self.peak_cpu_mem_logs) > 0:
            print("Dumping cpu leftover measures")
            self._log_cpu_mem()


        if len(self.loss_logs) > 0:
            print("Dumping loss leftover measures")
            self._log_loss()

        
    def _log_gpu_mem(self):
        if os.path.exists(self.peak_gpu_mem_logs_path):
            os.makedirs(self.peak_gpu_mem_logs_path, exist_ok=True)

        prev_peak_gpu_mem_logs_path = os.path.join(self.peak_gpu_mem_logs_path, self.lbl + "_peak_gpu_mem_logs.npz")
        prev_peak_gpu_mem_logs = np.load(prev_peak_gpu_mem_logs_path)["arr_0"] if os.path.exists(prev_peak_gpu_mem_logs_path) else np.array([])

        final_peak_gpu_mem_logs = np.append(prev_peak_gpu_mem_logs, np.array(self.peak_gpu_mem_logs))
        np.savez_compressed(os.path.join(self.peak_gpu_mem_logs_path, self.lbl + "_peak_gpu_mem_logs.npz"), final_peak_gpu_mem_logs)

        self.peak_gpu_mem_logs = []

        
        prev_peak_gpu_mem_timestamps_path = os.path.join(self.peak_gpu_mem_logs_path, self.lbl + "_peak_gpu_mem_timestamps.npz")
        prev_peak_gpu_mem_timestamps = np.load(prev_peak_gpu_mem_timestamps_path)["arr_0"] if os.path.exists(prev_peak_gpu_mem_timestamps_path) else np.array([])

        final_peak_gpu_mem_timestamps = np.append(prev_peak_gpu_mem_timestamps, np.array(self.peak_gpu_mem_timestamps))
        np.savez_compressed(os.path.join(self.peak_gpu_mem_logs_path, self.lbl + "_peak_gpu_mem_timestamps.npz"), final_peak_gpu_mem_timestamps)

        self.peak_gpu_mem_timestamps = []

    def _log_cpu_mem(self):
        if os.path.exists(self.peak_cpu_mem_logs_path):
            os.makedirs(self.peak_cpu_mem_logs_path, exist_ok=True)

        prev_peak_cpu_mem_logs_path = os.path.join(self.peak_cpu_mem_logs_path, self.lbl + "_peak_cpu_mem_logs.npz")
        prev_peak_cpu_mem_logs = np.load(prev_peak_cpu_mem_logs_path)["arr_0"] if os.path.exists(prev_peak_cpu_mem_logs_path) else np.array([])

        final_peak_cpu_mem_logs = np.append(prev_peak_cpu_mem_logs, np.array(self.peak_cpu_mem_logs))
        np.savez_compressed(os.path.join(self.peak_cpu_mem_logs_path, self.lbl + "_peak_cpu_mem_logs.npz"), final_peak_cpu_mem_logs)

        self.peak_cpu_mem_logs = []

        prev_peak_cpu_mem_timestamps_path = os.path.join(self.peak_cpu_mem_logs_path, self.lbl + "_peak_cpu_mem_timestamps.npz")
        prev_peak_cpu_mem_timestamps = np.load(prev_peak_cpu_mem_timestamps_path)["arr_0"] if os.path.exists(prev_peak_cpu_mem_timestamps_path) else np.array([])

        final_peak_cpu_mem_timestamps = np.append(prev_peak_cpu_mem_timestamps, np.array(self.peak_cpu_mem_timestamps))
        np.savez_compressed(os.path.join(self.peak_cpu_mem_logs_path, self.lbl + "_peak_cpu_mem_timestamps.npz"), final_peak_cpu_mem_timestamps)

        self.peak_cpu_mem_timestamps = []

    def _log_loss(self):
        if os.path.exists(self.loss_logs_path):
            os.makedirs(self.loss_logs_path, exist_ok=True)

        prev_loss_logs_path = os.path.join(self.loss_logs_path, self.lbl + "_loss_logs.npz")
        prev_loss_logs = np.load(prev_loss_logs_path)["arr_0"] if os.path.exists(prev_loss_logs_path) else np.array([])

        final_loss_logs = np.append(prev_loss_logs, np.array(self.loss_logs))
        np.savez_compressed(os.path.join(self.loss_logs_path, self.lbl + "_loss_logs.npz"), final_loss_logs)

        self.loss_logs = []

        prev_loss_timestamps_path = os.path.join(self.loss_logs_path, self.lbl + "_loss_timestamps.npz")
        prev_loss_timestamps = np.load(prev_loss_timestamps_path)["arr_0"] if os.path.exists(prev_loss_timestamps_path) else np.array([])

        final_loss_timestamps = np.append(prev_loss_timestamps, np.array(self.loss_timestamps))
        np.savez_compressed(os.path.join(self.loss_logs_path, self.lbl + "_loss_timestamps.npz"), final_loss_timestamps)

        self.loss_timestamps = []