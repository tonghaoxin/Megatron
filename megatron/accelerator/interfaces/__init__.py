# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""Interface Manager"""

from megatron.accelerator.interfaces.abstract_interface_manager import MegatronInterfaceManager
from megatron.accelerator.interfaces.cuda_interface_manager import CUDAInterfaceManager
from megatron.accelerator.interfaces.xpu_interface_manager import XPUInterfaceManager
