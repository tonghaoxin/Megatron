# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""MegatronAccelerator."""

import abc
from abc import ABC


class MegatronAccelerator(ABC):
    """MegatronAccelerator"""

    def __init__(self):
        """__init__"""
        self._backend_name = None
        self._communication_backend_name = None
        self._optimizers = None
        self._custom_operator_manager = None

    @abc.abstractmethod
    def backend_name(self):
        """backend_name"""
        ...

    # Device APIs
    @abc.abstractmethod
    def device_name(self, device_index):
        """device_name"""
        ...

    @abc.abstractmethod
    def device(self, device_index):
        """device"""
        ...

    @abc.abstractmethod
    def set_device(self, device_index):
        """set_device"""
        ...

    @abc.abstractmethod
    def current_device(self):
        """current_device"""
        ...

    @abc.abstractmethod
    def current_device_name(self):
        """current_device_name"""
        ...

    @abc.abstractmethod
    def device_count(self):
        """device_count"""
        ...

    @abc.abstractmethod
    def synchronize(self):
        """synchronize"""
        ...

    # RNG APIs
    @abc.abstractmethod
    def random(self):
        """random"""
        ...

    @abc.abstractmethod
    def set_rng_state(self, new_state, device_index=None):
        """set_rng_state"""
        ...

    @abc.abstractmethod
    def get_rng_state(self, device_index=None):
        """get_rng_state"""
        ...

    @abc.abstractmethod
    def manual_seed(self, seed):
        """manual_seed"""
        ...

    @abc.abstractmethod
    def manual_seed_all(self, seed):
        """manual_seed_all"""
        ...

    @abc.abstractmethod
    def initial_seed(self, seed):
        """initial_seed"""
        ...

    @abc.abstractmethod
    def default_generator(self, device_index):
        """default_generator"""
        ...

    @abc.abstractmethod
    def _set_rng_state(self, new_state, device_index=-1):
        """_set_rng_state"""
        ...

    # Streams/Events
    @property
    @abc.abstractmethod
    def Stream(self):
        """Stream"""
        ...

    @abc.abstractmethod
    def stream(self, stream):
        """stream"""
        ...

    @abc.abstractmethod
    def current_stream(self, device_index=None):
        """current_stream"""
        ...

    @abc.abstractmethod
    def default_stream(self, device_index=None):
        """default_stream"""
        ...

    @property
    @abc.abstractmethod
    def Event(self):
        """Event"""
        ...

    # Memory management
    @abc.abstractmethod
    def empty_cache(self):
        """empty_cache"""
        ...

    @abc.abstractmethod
    def memory_allocated(self, device_index=None):
        """memory_allocated"""
        ...

    @abc.abstractmethod
    def max_memory_allocated(self, device_index=None):
        """max_memory_allocated"""
        ...

    @abc.abstractmethod
    def reset_max_memory_allocated(self, device_index=None):
        """reset_max_memory_allocated"""
        ...

    @abc.abstractmethod
    def memory_cached(self, device_index=None):
        """memory_cached"""
        ...

    @abc.abstractmethod
    def max_memory_cached(self, device_index=None):
        """max_memory_cached"""
        ...

    @abc.abstractmethod
    def reset_max_memory_cached(self, device_index=None):
        """reset_max_memory_cached"""
        ...

    @abc.abstractmethod
    def memory_stats(self, device_index=None):
        """memory_stats"""
        ...

    @abc.abstractmethod
    def reset_peak_memory_stats(self, device_index=None):
        """reset_peak_memory_stats"""
        ...

    @abc.abstractmethod
    def memory_reserved(self, device_index=None):
        """memory_reserved"""
        ...

    @abc.abstractmethod
    def max_memory_reserved(self, device_index=None):
        """max_memory_reserved"""
        ...

    @abc.abstractmethod
    def total_memory(self, device_index=None):
        """total_memory"""
        ...

    # Data types
    @abc.abstractmethod
    def autocast(self, dtype):
        """autocast"""
        ...

    @abc.abstractmethod
    def is_bf16_supported(self):
        """is_bf16_supported"""
        ...

    @abc.abstractmethod
    def is_fp16_supported(self):
        """is_fp16_supported"""
        ...

    @abc.abstractmethod
    def supported_dtypes(self):
        """supported_dtypes"""
        ...

    # Misc
    @abc.abstractmethod
    def amp(self):
        """amp"""
        ...

    @abc.abstractmethod
    def is_available(self):
        """is_available"""
        ...

    @abc.abstractmethod
    def range_push(self, msg):
        """range_push"""
        ...

    @abc.abstractmethod
    def range_pop(self):
        """range_pop"""
        ...

    @abc.abstractmethod
    def get_fwd_profiler_ctx(self):
        """get_fwd_profiler_ctx"""
        ...

    @abc.abstractmethod
    def get_bwd_profiler_ctx(self):
        """get_bwd_profiler_ctx"""
        ...

    @abc.abstractmethod
    def profiler_start(self, profiler_ctx):
        """profiler_start"""
        ...

    @abc.abstractmethod
    def profiler_stop(self, profiler_ctx):
        """profiler_stop"""
        ...

    @abc.abstractmethod
    def get_torch_profiler(self, schedule, tensorboard_trace_handler,
                           record_shapes=False,
                           profile_memory=False,
                           with_stack=False,
                           with_flops=False,
                           with_modules=False):
        """get_torch_profiler"""
        ...

    @abc.abstractmethod
    def lazy_call(self, callback):
        """lazy_call"""
        ...

    @abc.abstractmethod
    def communication_backend_name(self):
        """communication_backend_name"""
        ...

    # Tensor operations
    @property
    @abc.abstractmethod
    def BFloat16Tensor(self):
        """BFloat16Tensor"""
        ...

    @property
    @abc.abstractmethod
    def ByteTensor(self):
        """ByteTensor"""
        ...

    @property
    @abc.abstractmethod
    def DoubleTensor(self):
        """DoubleTensor"""
        ...

    @property
    @abc.abstractmethod
    def FloatTensor(self):
        """FloatTensor"""
        ...

    @property
    @abc.abstractmethod
    def HalfTensor(self):
        """HalfTensor"""
        ...

    @property
    @abc.abstractmethod
    def IntTensor(self):
        """IntTensor"""
        ...

    @property
    @abc.abstractmethod
    def LongTensor(self):
        """LongTensor"""
        ...

    @abc.abstractmethod
    def BFloat16Tensor_name(self):
        """BFloat16Tensor_name"""
        ...

    @abc.abstractmethod
    def ByteTensor_name(self):
        """ByteTensor_name"""
        ...

    @abc.abstractmethod
    def DoubleTensor_name(self):
        """DoubleTensor_name"""
        ...

    @abc.abstractmethod
    def FloatTensor_name(self):
        """FloatTensor_name"""
        ...

    @abc.abstractmethod
    def HalfTensor_name(self):
        """HalfTensor_name"""
        ...

    @abc.abstractmethod
    def IntTensor_name(self):
        """IntTensor_name"""
        ...

    @abc.abstractmethod
    def LongTensor_name(self):
        """LongTensor_name"""
        ...

    @abc.abstractmethod
    def pin_memory(self, tensor):
        """pin_memory"""
        ...

    @abc.abstractmethod
    def h2d_copy(self, obj):
        """h2d_copy"""
        ...

    @abc.abstractmethod
    def d2h_copy(self, obj):
        """d2h_copy"""
        ...

    @abc.abstractmethod
    def on_accelerator(self, tensor):
        """on_accelerator"""
        ...

    @abc.abstractmethod
    def is_contiguous(self, tensor):
        """is_contiguous"""
        ...

    @abc.abstractmethod
    def load_custom_ops(self):
        """load_custom_ops"""
        ...

    @abc.abstractmethod
    def get_operator(self, op_name):
        """get_operator"""
        ...

    @abc.abstractmethod
    def get_optimizer(self, optim_name):
        """get_optimizer"""
        ...
