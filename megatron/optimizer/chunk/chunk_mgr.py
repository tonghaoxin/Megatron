"""
    Manager for Chunk
"""
import torch
from collections import deque
from megatron.optimizer.chunk.chunk import Chunk
from megatron.accelerator import get_accelerator
from typing import Deque, Dict, Optional

class ChunkMgr:
    """
    A manager class to manipulate the tensors in chunks.
    """

    def __init__(self, init_device: Optional[torch.device] = None) -> None:
        """
        A manager class to manipulate the tensors in chunks.

        Args:
            chunk_configuration (Dict[int, Dict]): the configuration dictionary of this chunk manager.
            init_device (torch.device): optional, the device on which the chunk is initialized. The default is None.
        """

        self.device = init_device or get_accelerator().current_device()

        self.chunk_groups: Dict[str, Deque] = dict()
        self.tensor_chunk_map: Dict[torch.Tensor, Chunk] = dict()
        self.total_mem: Dict[str, int] = {'cpu': 0, 'cuda': 0}

    def register_tensor(self,
                        tensor: torch.Tensor,
                        group_type: str,
                        device: torch.device,
                        pin_memory: bool = False) -> None:
        """
        Register a tensor to the chunk manager.

        Args:
            tensor: the tensor appended to the chunk
            group_type: the data type of the group.
            device: the device for tensor.
            pin_memory: whether the chunk is pinned in the cpu memory
        """
        assert tensor not in self.tensor_chunk_map

        chunk_size = tensor.numel()
        group_name = group_type
        chunk_group = self.__get_chunk_group(group_name)

        chunk = Chunk(
            chunk_size=chunk_size,
            dtype=tensor.dtype,
            device=device,
            pin_memory=pin_memory
        )

        chunk_group.append(chunk)
        chunk.append_tensor(tensor)
        self.__add_memory_usage(chunk.memory_usage)

        self.tensor_chunk_map[tensor] = chunk

    def close_all_groups(self):
        """Close all the chunks of all groups.
        """
        for group_name in self.chunk_groups:
            self.__close_one_chunk(self.chunk_groups[group_name][-1])

    def __get_chunk_group(self, group_name: str) -> Deque:
        """Register a chunk group.
        """
        if group_name not in self.chunk_groups:
            self.chunk_groups[group_name] = deque()
        return self.chunk_groups[group_name]

    def __close_one_chunk(self, chunk: Chunk):
        """__close_one_chunk"""
        self.__sub_memory_usage(chunk.memory_usage)
        chunk.close_chunk()
        self.__add_memory_usage(chunk.memory_usage)

    def __sub_memory_usage(self, usage: Dict[str, int]):
        """__sub_memory_usage"""
        for k, v in usage.items():
            self.total_mem[k] -= v

    def __add_memory_usage(self, usage: Dict[str, int]):
        """__add_memory_usage"""
        for k, v in usage.items():
            self.total_mem[k] += v

