"""
    Memory status
"""
from enum import Enum

from megatron import get_args

class MemoryType(Enum):
    """
    Memory type for record
    """
    MODEL_DATA_CUDA = 1
    NON_MODEL_DATA_CUDA = 2
    OPTIMIZER_BASE_CUDA = 3

class WarmupStage(Enum):
    """
    The stage of warmup, DOING_WARMUP -> FINISHED
    """
    DOING_WARMUP = 1
    FINISHED = 3

class MemStats(object):
    """
    Class used to hold memory status
    """

    def __init__(self):
        """
        Store the gpu and cpu used for DistributeOptimizer and ZeroOptimizer.
        """
        args = get_args()
        if args.cpu_offload == 'auto':
            self._warmup_stage = WarmupStage.DOING_WARMUP
        else:
            self._warmup_stage = WarmupStage.FINISHED
        self._model_data_cuda_list = []
        self._non_model_data_cuda_list = []

        self._optimizer_base_cuda_list = []
        self._warmup_steps = 0

    def record_memory(self, memory_type, val):
        """
        Record the memory type of memory_type, and value is val.
        """
        if memory_type == MemoryType.MODEL_DATA_CUDA:
            self._model_data_cuda_list.append(val)
        elif memory_type == MemoryType.NON_MODEL_DATA_CUDA:
            self._non_model_data_cuda_list.append(val)
        elif memory_type == MemoryType.OPTIMIZER_BASE_CUDA:
            self._optimizer_base_cuda_list.append(val)
        pass

    def get_memory(self, memory_type):
        """
        Get the memory by memory_type.
        """
        if memory_type == MemoryType.MODEL_DATA_CUDA:
            return max(self._model_data_cuda_list)
        elif memory_type == MemoryType.NON_MODEL_DATA_CUDA:
            return max(self._non_model_data_cuda_list)
        elif memory_type == MemoryType.OPTIMIZER_BASE_CUDA:
            return max(self._optimizer_base_cuda_list)
        return 0

    @property
    def is_warmup(self):
        """
        The optimizer is doing warmup.
        """
        return self._warmup_stage == WarmupStage.DOING_WARMUP

    def increase_warmup_steps(self):
        """
        self._warmup_steps +1, and check whethor self._warmup_stage will be changed.
        """
        args = get_args()
        self._warmup_steps += 1
        if self._warmup_stage == WarmupStage.DOING_WARMUP and self._warmup_steps > 0:
            self._warmup_stage = WarmupStage.FINISHED

    def clear(self):
        """
        clear all data in memory status.
        """
        self._model_data_cuda_list = []
        self._non_model_data_cuda_list = []
        self._optimizer_base_cuda_list = []

