# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

import os

from .custom_ops import noop_decorator

try:
    from megatron.accelerator.abstract_accelerator import MegatronAccelerator as accel
except ImportError as e:
    accel = None

accelerator = None


def _validate_accelerator(accel_obj):
    """_validate_accelerator"""
    if not (accel is not None and isinstance(accel_obj, accel)):
        raise AssertionError(f"{accel_obj.__class__.__name__} accelerator is not subclass of MegatronAccelerator")


def get_accelerator():
    """get_accelerator"""
    global accelerator
    if accelerator is not None:
        return accelerator

    MEGATRON_ACCELERATOR_LIST = ["cuda", "xpu"]
    accelerator_name = None

    # Check the accelerator name from environment variable
    if "MEGATRON_ACCELERATOR" in os.environ.keys():
        accelerator_name = os.environ["MEGATRON_ACCELERATOR"]

    # Auto detection if the accelerator name is not provided
    if accelerator_name is None:
        try:
            import torch_xmlir
            accelerator_name = "xpu"
        except ImportError as e:
            pass
        if accelerator_name is None:
            accelerator_name = "cuda"

    if accelerator_name == "cuda":
        from .cuda_accelerator import CUDAAccelerator
        accelerator = CUDAAccelerator()
    elif accelerator_name == "xpu":
        from .xpu_accelerator import XPUAccelerator
        accelerator = XPUAccelerator()
    else:
        raise ValueError(
            f"MEGATRON_ACCELERATOR must be one of {MEGATRON_ACCELERATOR_LIST}.\
            Value {accelerator_name} is not supported")

    _validate_accelerator(accelerator)

    print(f'> setting accelerator backend to {accelerator_name} ...')

    return accelerator
