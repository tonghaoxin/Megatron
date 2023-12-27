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
        self.total_mem: Dict[str, Dict[str, Deque]] = dict()

    def register_tensor(self,
                        tensor: torch.Tensor,
                        group_name: str,
                        device: torch.device) -> None:
        """
        Register a tensor to the chunk manager.

        Args:
            tensor: the tensor appended to the chunk
            group_name: the data type of the group.
            device: the device for tensor.
        """
        assert tensor not in self.tensor_chunk_map

        chunk_size = tensor.numel()
        chunk_group = self.__get_chunk_group(group_name)
        if group_name not in self.total_mem:
            self.total_mem[group_name] = {'cpu': 0, 'cuda': 0}

        chunk = Chunk(
            chunk_size=chunk_size,
            dtype=tensor.dtype,
            device=device
        )

        chunk_group.append(chunk)
        self.__add_memory_usage(group_name, chunk.memory_usage)
        chunk.reset_tensor(tensor)

        self.tensor_chunk_map[tensor] = chunk

    def reset_tensor(self,
                     group_name,
                     tensor: torch.Tensor,
                     device: torch.device) -> None:
        """
        Reset a tensor to the chunk.

        Args:
            group_name: the data type of the group.
            tensor: the tensor appended to the chunk
            device: the device for tensor.
        """
        assert tensor in self.tensor_chunk_map

        chunk = self.tensor_chunk_map[tensor]
        self.__sub_memory_usage(group_name, chunk.memory_usage)
        chunk.reset_chunk_tensor(tensor.numel(), tensor.dtype, device)
        chunk.reset_tensor(tensor)
        self.__add_memory_usage(group_name, chunk.memory_usage)

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

    def __close_one_chunk(self, group_name: str, chunk: Chunk):
        """__close_one_chunk"""
        assert group_name in self.total_mem, f"No key in total_mem group_name: {group_name}"
        self.__sub_memory_usage(group_name, chunk.memory_usage)
        chunk.close_chunk()
        self.__add_memory_usage(group_name, chunk.memory_usage)

    def __sub_memory_usage(self, group_name: str, usage: Dict[str, int]):
        """__sub_memory_usage"""
        assert group_name in self.total_mem, f"No key in total_mem group_name: {group_name}"
        for k, v in usage.items():
            self.total_mem[group_name][k] -= v

    def __add_memory_usage(self, group_name: str, usage: Dict[str, int]):
        """__add_memory_usage"""
        assert group_name in self.total_mem, f"No key in total_mem group_name: {group_name}"
        for k, v in usage.items():
            self.total_mem[group_name][k] += v

    def memory_usage(self, group_name: str) -> Dict[str, int]:
        """ The memory usage for chunk
        """
        if group_name in self.chunk_groups:
            return self.total_mem[group_name]
        else:
            return {'cpu': 0, 'cuda': 0}
