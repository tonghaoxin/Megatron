"""
    Chunk used for tensor block
"""
from typing import Dict

import torch

class Chunk:
    """
    Chunk: A container owning a piece of contiguous memory space for tensors
    """
    def __init__(self,
                 chunk_size: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 pin_memory: bool = False) -> None:
        """
        Chunk: A container owning a piece of contiguous memory space for tensors

        Args:
            chunk_size (int): the number of elements in the chunk
            dtype (torch.dtype): the data type of the chunk
            device (torch.device): the device for tensor.
            pin_memory (bool): optional, if True, this chunk always has a shard copied in pinned CPU memory
        """

        assert chunk_size > 0
        assert dtype is not None
        assert device is not None

        self.chunk_size = chunk_size
        self.dtype = dtype
        self.device = device
        self.pin_memory = pin_memory

        # chunk_tensor is a global chunk, which only exists during building the chunks.
        self.chunk_tensor = torch.zeros(chunk_size, dtype=dtype, device=device, pin_memory=self.pin_memory)
        if device == torch.device('cpu'):
            self.chunk_tensor.grad = torch.zeros(chunk_size, dtype=dtype, device=device, pin_memory=self.pin_memory)

        self.chunk_mem = self.chunk_size * self.chunk_tensor.element_size()

    def append_tensor(self, tensor: torch.Tensor):
        """Add a tensor to the chunk.

        Args:
            tensor (torch.Tensor): a tensor to be added to the chunk
        """
        # sanity check
        assert self.chunk_tensor is not None
        assert tensor.dtype == self.dtype

        self.chunk_tensor.copy_(tensor.data.flatten())
        tensor.data = self.chunk_tensor.view(tensor.shape)
        if self.device == torch.device('cpu'):
            tensor.grad = self.chunk_tensor.grad.view(tensor.shape)

    def close_chunk(self):
        """Close the chunk. Any tensor can't be appended to a closed chunk later.
        """
        # sanity check
        assert self.chunk_tensor is not None

        self.chunk_tensor = None

    @property
    def memory_usage(self) -> Dict[str, int]:
        """
            Dict for usage of memory
        """
        cuda_memory = 0
        cpu_memory = 0

        if self.chunk_tensor is not None:
            # this chunk is not closed
            if self.chunk_tensor.device.type == 'cuda':
                cuda_memory += self.chunk_mem
            else:
                cpu_memory += self.chunk_mem

        return dict(cuda=cuda_memory, cpu=cpu_memory)

