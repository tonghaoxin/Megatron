# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""NPUAccelerator."""

import torch

from .abstract_accelerator import MegatronAccelerator
from .custom_ops import NPUCustomOperatorManager

# During setup stage torch may not be installed, pass on no torch will
# allow op builder related API to be executed.
try:
    import torch_npu
    from torch_npu import _C
except ImportError:
    pass


class NPUAccelerator(MegatronAccelerator):
    """NPUAccelerator"""

    def __init__(self):
        """__init__"""
        self._backend_name = 'npu'
        self._communication_backend_name = 'hccl'
        self._optimizers = {}

        try:
            from .npu_adam import AdamW as adam_optimizer_npu
            from torch.optim import SGD as sgd_optimizer_npu
            #from torch.optim import AdamW as adam_optimizer_npu
            # from apex.optimizers import NpuFusedAdam as adam_optimizer_npu
            # from apex.optimizers import NpuFusedSGD as sgd_optimizer_npu
        except ImportError:
            raise RuntimeError(
                "Failed to import optimizers."
            )

        self._optimizers['adam'] = adam_optimizer_npu
        self._optimizers['sgd'] = sgd_optimizer_npu

        self._custom_operator_manager = NPUCustomOperatorManager()

    def backend_name(self):
        """backend_name"""
        return self._backend_name

    # Device APIs
    def device_name(self, device_index=None):
        """device_name"""
        if device_index is None:
            return 'npu'
        return 'npu:{}'.format(device_index)

    def device(self, device_index=None):
        """device"""
        return torch.cuda.device(device_index)

    def set_device(self, device_index):
        """set_device"""
        torch.cuda.set_device(device_index)

    def current_device(self):
        """current_device"""
        return torch.cuda.current_device()

    def current_device_name(self):
        """current_device_name"""
        return 'cuda:{}'.format(torch.cuda.current_device())

    def device_count(self):
        """device_count"""
        return torch.cuda.device_count()

    def synchronize(self):
        """synchronize"""
        return torch.cuda.synchronize()

    # RNG APIs
    def random(self):
        """random"""
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        """set_rng_state"""
        if device_index is None:
            return torch.cuda.set_rng_state(new_state)

        return torch.cuda.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        """get_rng_state"""
        if device_index is None:
            return torch.cuda.get_rng_state()

        return torch.cuda.get_rng_state(device_index)

    def manual_seed(self, seed):
        """manual_seed"""
        return torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed):
        """manual_seed_all"""
        return torch.cuda.manual_seed_all(seed)

    def initial_seed(self, seed):
        """initial_seed"""
        return torch.cuda.initial_seed(seed)

    def default_generator(self, device_index):
        """default_generator"""
        return torch.cuda.default_generators[device_index]

    def _set_rng_state(self, new_state, device_index=-1):
        """Sets the random number generator state of the current GPU.

        Argumentss:
            new_state (torch.ByteTensor): The desired state
            device_index (int | str): Device index
        This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
        with a single change: the input state is not cloned. Cloning caused
        major performance issues for +4 GPU cases.
        """
        if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
            # older PyTorch
            def cb():
                """cb"""
                with torch.cuda.device(device_index):
                    _C._cuda_setRNGState(new_state)
        else:
            # newer PyTorch
            if device_index == -1:
                device = torch.device('cuda')
            elif isinstance(device_index, str):
                device = torch.device(device_index)
            elif isinstance(device_index, int):
                device = torch.device('cuda', device_index)

            def cb():
                """cb"""
                idx = device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                # TODO: find the reason that we can't use torch.cuda.default_generators
                default_generator = torch.npu.default_generators[idx]
                default_generator.set_state(new_state)

        torch.cuda._lazy_call(cb)

    # Streams/Events
    @property
    def Stream(self):
        """Stream"""
        return torch.cuda.Stream

    def stream(self, stream):
        """stream"""
        return torch.cuda.stream(stream)

    def current_stream(self, device_index=None):
        """current_stream"""
        return torch.cuda.current_stream(device_index)

    def default_stream(self, device_index=None):
        """default_stream"""
        return torch.cuda.default_stream(device_index)

    @property
    def Event(self):
        """Event"""
        return torch.cuda.Event

    # Memory management
    def empty_cache(self):
        """empty_cache"""
        return torch.cuda.empty_cache()

    def memory_allocated(self, device_index=None):
        """memory_allocated"""
        return torch.cuda.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        """max_memory_allocated"""
        return torch.cuda.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        """reset_max_memory_allocated"""
        return torch.cuda.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        """memory_cached"""
        return torch.cuda.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        """max_memory_cached"""
        return torch.cuda.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        """reset_max_memory_cached"""
        return torch.cuda.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        """memory_stats"""
        if hasattr(torch.cuda, 'memory_stats'):
            return torch.cuda.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        """reset_peak_memory_stats"""
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            return torch.cuda.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        """memory_reserved"""
        if hasattr(torch.cuda, 'memory_reserved'):
            return torch.cuda.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        """max_memory_reserved"""
        if hasattr(torch.cuda, 'max_memory_reserved'):
            return torch.cuda.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        """total_memory"""
        return torch.cuda.get_device_properties(device_index).total_memory

    # Data types
    def autocast(self, dtype):
        """autocast"""
        torch.autocast("cuda", dtype=dtype)

    def is_bf16_supported(self):
        """is_bf16_supported"""
        return torch.cuda.is_bf16_supported()

    def is_fp16_supported(self):
        """is_fp16_supported"""
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            return True
        else:
            return False

    def supported_dtypes(self):
        """supported_dtypes"""
        return [torch.float, torch.half, torch.bfloat16]

    # Misc
    def amp(self):
        """amp"""
        if hasattr(torch.cuda, 'amp'):
            return torch.cuda.amp
        return None

    def is_available(self):
        """is_available"""
        return torch.cuda.is_available()

    def range_push(self, msg):
        """range_push"""
        if hasattr(torch.cuda.nvtx, 'range_push'):
            return torch.cuda.nvtx.range_push(msg)

    def range_pop(self):
        """range_pop"""
        if hasattr(torch.cuda.nvtx, 'range_pop'):
            return torch.cuda.nvtx.range_pop()

    def get_fwd_profiler_ctx(self):
        """get_fwd_profiler_ctx"""
        return None

    def get_bwd_profiler_ctx(self):
        """get_bwd_profiler_ctx"""
        return None

    def profiler_start(self, profiler_ctx):
        """profiler_start"""
        pass

    def profiler_stop(self, profiler_ctx):
        """profiler_stop"""
        pass

    def get_torch_profiler(self, schedule, tensorboard_trace_handler,
                           record_shapes=False,
                           profile_memory=False,
                           with_stack=False,
                           with_flops=False,
                           with_modules=False):
        """get_torch_profiler"""
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ]

        experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
                l2_cache=False,
                record_op_args=True,
                data_simplification=False
        )

        return torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=tensorboard_trace_handler,
                record_shapes=record_shapes,
                profile_memory=profile_memory,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules,
                experimental_config=experimental_config)

    def lazy_call(self, callback):
        """lazy_call"""
        return torch.cuda._lazy_call(callback)

    def communication_backend_name(self):
        """communication_backend_name"""
        return self._communication_backend_name

    # Tensor operations
    @property
    def BFloat16Tensor(self):
        """BFloat16Tensor"""
        return torch.cuda.BFloat16Tensor

    @property
    def ByteTensor(self):
        """ByteTensor"""
        return torch.cuda.ByteTensor

    @property
    def DoubleTensor(self):
        """DoubleTensor"""
        return torch.cuda.DoubleTensor

    @property
    def FloatTensor(self):
        """FloatTensor"""
        return torch.cuda.FloatTensor

    @property
    def HalfTensor(self):
        """HalfTensor"""
        return torch.cuda.HalfTensor

    @property
    def IntTensor(self):
        """IntTensor"""
        return torch.cuda.IntTensor

    @property
    def LongTensor(self):
        """LongTensor"""
        return torch.cuda.LongTensor

    def BFloat16Tensor_name(self):
        """BFloat16Tensor_name"""
        return 'torch.npu.BFloat16Tensor'

    def ByteTensor_name(self):
        """ByteTensor_name"""
        return 'torch.npu.ByteTensor'

    def DoubleTensor_name(self):
        """DoubleTensor_name"""
        return 'torch.npu.DoubleTensor'

    def FloatTensor_name(self):
        """FloatTensor_name"""
        return 'torch.npu.FloatTensor'

    def HalfTensor_name(self):
        """HalfTensor_name"""
        return 'torch.npu.HalfTensor'

    def IntTensor_name(self):
        """IntTensor_name"""
        return 'torch.npu.IntTensor'

    def LongTensor_name(self):
        """LongTensor_name"""
        return 'torch.npu.LongTensor'

    def pin_memory(self, tensor):
        """pin_memory"""
        return tensor.pin_memory()

    def h2d_copy(self, obj, device=None):
        """h2d_copy"""
        if obj is None:
            return
        if device is None:
            device = torch.cuda.current_device()
        return obj.cuda(device)

    def d2h_copy(self, obj):
        """d2h_copy"""
        if obj is None:
            return
        return obj.cpu()

    def on_accelerator(self, tensor):
        """on_accelerator"""
        return tensor.is_cuda

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
