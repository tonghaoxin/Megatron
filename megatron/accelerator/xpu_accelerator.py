# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""XPUAccelerator."""

import os
import pkgutil
import importlib
import torch

from torch import _C
from .abstract_accelerator import MegatronAccelerator
from .custom_ops import XPUCustomOperatorManager

try:
    import torch_xmlir
except ImportError:
    pass


class XPUAccelerator(MegatronAccelerator):
    """XPUAccelerator"""

    def __init__(self):
        """__init__"""
        self._communication_backend_name = 'nccl'
        self._optimizers = {}

        try:
            from torch_xmlir.optimizer.fused_adamw import FusedAdamW as adam_optimizer
            from torch.optim import SGD as sgd_optimizer
        except ImportError:
            raise RuntimeError(
                "Failed to import optimizers."
            )

        self._optimizers['adam'] = adam_optimizer
        self._optimizers['sgd'] = sgd_optimizer

        self._custom_operator_manager = XPUCustomOperatorManager()

    # Device APIs
    def device_name(self, device_index=None):
        """device_name"""
        if device_index is None:
            return 'xpu'
        return 'xpu:{}'.format(device_index)

    def device(self, device_index=None):
        """device"""
        return torch_xmlir.xpu.device(device_index)

    def set_device(self, device_index):
        """set_device"""
        torch_xmlir.xpu.set_device(device_index)

    def current_device(self):
        """current_device"""
        return torch_xmlir.xpu.current_device()

    def current_device_name(self):
        """current_device_name"""
        return 'xpu:{}'.format(torch_xmlir.xpu.current_device())

    def device_count(self):
        """device_count"""
        return torch_xmlir.xpu.device_count()

    def synchronize(self):
        """synchronize"""
        return torch_xmlir.xpu.synchronize()

    # RNG APIs
    def random(self):
        """random"""
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        """set_rng_state"""
        if device_index is None:
            return torch_xmlir.xpu.set_rng_state(new_state)

        return torch_xmlir.xpu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        """get_rng_state"""
        if device_index is None:
            return torch_xmlir.xpu.get_rng_state()

        return torch_xmlir.xpu.get_rng_state(device_index)

    def manual_seed(self, seed):
        """manual_seed"""
        return torch_xmlir.xpu.manual_seed(seed)

    def manual_seed_all(self, seed):
        """manual_seed_all"""
        return torch_xmlir.xpu.manual_seed_all(seed)

    def initial_seed(self, seed):
        """initial_seed"""
        return torch_xmlir.xpu.initial_seed(seed)

    def default_generator(self, device_index):
        """default_generator"""
        return torch_xmlir.xpu.default_generators[device_index]

    def _set_rng_state(self, new_state, device_index=-1):
        """Sets the random number generator state of the current GPU.

        Argumentss:
            new_state (torch.ByteTensor): The desired state
            device_index (int | str): Device index
        This function is adapted from PyTorch repo (torch_xmlir.xpu.set_rng_state)
        with a single change: the input state is not cloned. Cloning caused
        major performance issues for +4 GPU cases.
        """
        if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
            # older PyTorch
            def cb():
                """cb"""
                with torch_xmlir.xpu.device(device_index):
                    _C._cuda_setRNGState(new_state)
        else:
            # newer PyTorch
            if device_index == -1:
                device = torch.device('xpu')
            elif isinstance(device_index, str):
                device = torch.device(device_index)
            elif isinstance(device_index, int):
                device = torch.device('xpu', device_index)

            def cb():
                """cb"""
                idx = device.index
                if idx is None:
                    idx = torch_xmlir.xpu.current_device()
                default_generator = torch_xmlir.xpu.default_generators[idx]
                default_generator.set_state(new_state)

        torch.cuda._lazy_call(cb)

    # Streams/Events
    @property
    def Stream(self):
        """Stream"""
        return torch_xmlir.xpu.Stream

    def stream(self, stream):
        """stream"""
        return torch_xmlir.xpu.stream(stream)

    def current_stream(self, device_index=None):
        """current_stream"""
        return torch_xmlir.xpu.current_stream(device_index)

    def default_stream(self, device_index=None):
        """default_stream"""
        return torch_xmlir.xpu.default_stream(device_index)

    @property
    def Event(self):
        """Event"""
        return torch_xmlir.xpu.Event

    # Memory management
    def empty_cache(self):
        """empty_cache"""
        return torch_xmlir.xpu.empty_cache()

    def memory_allocated(self, device_index=None):
        """memory_allocated"""
        return torch_xmlir.xpu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        """max_memory_allocated"""
        return torch_xmlir.xpu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        """reset_max_memory_allocated"""
        return torch_xmlir.xpu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        """memory_cached"""
        return torch_xmlir.xpu.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        """max_memory_cached"""
        return torch_xmlir.xpu.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        """reset_max_memory_cached"""
        return torch_xmlir.xpu.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        """memory_stats"""
        if hasattr(torch_xmlir.xpu, 'memory_stats'):
            return torch_xmlir.xpu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        """reset_peak_memory_stats"""
        if hasattr(torch_xmlir.xpu, 'reset_peak_memory_stats'):
            return torch_xmlir.xpu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        """memory_reserved"""
        if hasattr(torch_xmlir.xpu, 'memory_reserved'):
            return torch_xmlir.xpu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        """max_memory_reserved"""
        if hasattr(torch_xmlir.xpu, 'max_memory_reserved'):
            return torch_xmlir.xpu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        """total_memory"""
        return torch_xmlir.xpu.get_device_properties(device_index).total_memory

    # Data types
    def autocast(self, dtype):
        """autocast"""
        torch.autocast("xpu", dtype=dtype)

    def is_bf16_supported(self):
        """is_bf16_supported"""
        return False

    def is_fp16_supported(self):
        """is_fp16_supported"""
        return True

    def supported_dtypes(self):
        """supported_dtypes"""
        return [torch.float, torch.half]

    # Misc
    def amp(self):
        """amp"""
        if hasattr(torch_xmlir.xpu, 'amp'):
            return torch_xmlir.xpu.amp
        return None

    def is_available(self):
        """is_available"""
        return torch_xmlir.xpu.is_available()

    def range_push(self, msg):
        """range_push"""
        if hasattr(torch_xmlir.xpu.nvtx, 'range_push'):
            return torch_xmlir.xpu.nvtx.range_push(msg)

    def range_pop(self):
        """range_pop"""
        if hasattr(torch_xmlir.xpu.nvtx, 'range_pop'):
            return torch_xmlir.xpu.nvtx.range_pop()

    def get_fwd_profiler_ctx(self):
        """get_fwd_profiler_ctx"""
        return None

    def get_bwd_profiler_ctx(self):
        """get_bwd_profiler_ctx"""
        return torch.autograd.profiler.emit_nvtx(record_shapes=True)

    def profiler_start(self, profiler_ctx):
        """profiler_start"""
        pass

    def profiler_stop(self, profiler_ctx):
        """profiler_stop"""
        pass

    def lazy_call(self, callback):
        """lazy_call"""
        return torch_xmlir.xpu._lazy_call(callback)

    def communication_backend_name(self):
        """communication_backend_name"""
        return self._communication_backend_name

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        """BFloat16Tensor"""
        return torch_xmlir.xpu.HalfTensor

    @property
    def ByteTensor(self):
        """ByteTensor"""
        return torch_xmlir.xpu.ByteTensor

    @property
    def DoubleTensor(self):
        """DoubleTensor"""
        return torch_xmlir.xpu.DoubleTensor

    @property
    def FloatTensor(self):
        """FloatTensor"""
        return torch_xmlir.xpu.FloatTensor

    @property
    def HalfTensor(self):
        """HalfTensor"""
        return torch_xmlir.xpu.HalfTensor

    @property
    def IntTensor(self):
        """IntTensor"""
        return torch_xmlir.xpu.IntTensor

    @property
    def LongTensor(self):
        """LongTensor"""
        return torch_xmlir.xpu.LongTensor

    def BFloat16Tensor_name(self):
        """BFloat16Tensor_name"""
        return 'torch.xpu.HalfTensor'

    def ByteTensor_name(self):
        """ByteTensor_name"""
        return 'torch.xpu.ByteTensor'

    def DoubleTensor_name(self):
        """DoubleTensor_name"""
        return 'torch.xpu.DoubleTensor'

    def FloatTensor_name(self):
        """FloatTensor_name"""
        return 'torch.xpu.FloatTensor'

    def HalfTensor_name(self):
        """HalfTensor_name"""
        return 'torch.xpu.HalfTensor'

    def IntTensor_name(self):
        """IntTensor_name"""
        return 'torch.xpu.IntTensor'

    def LongTensor_name(self):
        """LongTensor_name"""
        return 'torch.xpu.LongTensor'

    def pin_memory(self, tensor):
        """pin_memory"""
        return tensor.pin_memory()

    def h2d_copy(self, obj, device=None):
        """h2d_copy"""
        if device is None:
            device = torch_xmlir.xpu.current_device()
        return obj.to(device)

    def d2h_copy(self, obj):
        """d2h_copy"""
        return obj.cpu()

    def on_accelerator(self, tensor):
        """on_accelerator"""
        return tensor.is_xpu

    def is_contiguous(self, tensor):
        """is_contiguous"""
        return tensor.is_contiguous()

    def load_custom_ops(self, args):
        """load_custom_ops"""
        self._custom_operator_manager.register_ops(args)

    def get_operator(self, op_name):
        """get_operator"""
        return self._custom_operator_manager.get_operator(op_name)

    def get_optimizer(self, optim_name):
        """get_optimizer"""
        return self._optimizers[optim_name]
