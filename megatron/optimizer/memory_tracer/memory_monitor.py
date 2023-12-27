"""
    Memory monitor used to get memory usage
"""
from abc import abstractmethod
from time import sleep, time
from .memory_stats import MemStats

import torch

class MemoryMonitor(object):
    """
    Base class for all types of memory monitor.
    """

    def __init__(self):
        self.mem_stats = MemStats()
        self.running = True

    @abstractmethod
    def start(self):
        """
        Start a monitor and reset the recorded data.
        """
        pass

    @abstractmethod
    def finish(self):
        """
        Get the monitored memory usage from function start().
        """
        pass

    def state_dict(self):
        """
        Get the memory status recorded.
        """
        return {
            "mem_stats": self.mem_stats,
        }

    @abstractmethod
    def cur_memory(self):
        """
        Get current memory of cuda.
        """
        pass

    def clear(self):
        """
        Clear the memory data recorded.
        """
        self.mem_stats.clear()

class SyncCudaMemoryMonitor(MemoryMonitor):
    """
    A synchronized cuda memory monitor.
    It only record the maximum allocated cuda memory from start point to finish point.
    """

    def __init__(self):
        super().__init__()

    def start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def finish(self):
        torch.cuda.synchronize()
        #max_usage = torch.cuda.max_memory_cached()
        max_usage = torch.cuda.max_memory_allocated()
        return max_usage

    def cur_memory(self):
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()

