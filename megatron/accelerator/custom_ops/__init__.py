# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""Custom Operator Manager"""

from megatron.accelerator.custom_ops.abstract_custom_operator_manager import MegatronCustomOperatorManager
from megatron.accelerator.custom_ops.cuda_custom_operator_manager import CUDACustomOperatorManager
from megatron.accelerator.custom_ops.xpu_custom_operator_manager import XPUCustomOperatorManager
from megatron.accelerator.custom_ops.npu_custom_operator_manager import NPUCustomOperatorManager

def noop_decorator(func):
    """noop_decorator"""
    return func
