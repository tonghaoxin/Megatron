# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""NPUCustomOperatorManager."""

import os
import pkgutil
import importlib
import torch

from .abstract_custom_operator_manager import MegatronCustomOperatorManager
from ..interfaces import NPUInterfaceManager


class NPUCustomOperatorManager(MegatronCustomOperatorManager):
    """NPUCustomOperatorManager"""

    def __init__(self):
        """__init__"""
        self._custom_ops = {}
        self._custom_interface_manager = NPUInterfaceManager()

        # Register pre-init custom ops
        try:
            from torch import bmm as bmm_npu
            from torch import baddbmm as baddbmm_npu
            from torch.nn import SyncBatchNorm as sync_batch_norm_npu
            from torch.nn import LayerNorm
            from torch_npu import npu_scaled_masked_softmax as scaled_masked_softmax_npu
            from torch_npu import npu_fusion_attention as flashattn_func_npu
            from torch_npu import npu_rotary_mul as rotary_mul_npu
            from torch_npu import npu_rms_norm as rms_norm_npu
            from einops import rearrange as rearrange_npu
        except ImportError:
            raise RuntimeError(
                "Failed to import pre-init custom ops."
            )

        self._custom_ops['custom_layer_norm'] = self._custom_interface_manager.\
                get_interface('custom_layer_norm').wrap(LayerNorm)
        self._custom_ops['custom_rms_norm'] = self._custom_interface_manager.\
                get_interface('custom_rms_norm').wrap(torch.nn.Module, rms_norm_npu)
        self._custom_ops['custom_bmm'] = bmm_npu
        self._custom_ops['custom_baddbmm'] = baddbmm_npu
        self._custom_ops['custom_sync_batch_norm'] = sync_batch_norm_npu
        self._custom_ops['custom_softmax'] = self._custom_interface_manager.\
                get_interface('custom_softmax')(scaled_masked_softmax_npu)
        self._custom_ops['custom_apply_rope_no_freqs'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_no_freqs')(rotary_mul_npu)
        self._custom_ops['custom_apply_rope_with_freqs'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_with_freqs')
        self._custom_ops['custom_apply_rope_with_idx'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_with_idx')
        self._custom_ops['custom_fc'] = self._custom_interface_manager.\
                get_interface('custom_fc')()
        self._custom_ops['custom_grad_accum'] = self._custom_interface_manager.\
                get_interface('custom_grad_accum')()
        self._custom_ops['custom_multi_tensor_applier'] = self._custom_interface_manager.\
                get_interface('custom_multi_tensor_applier')()
        self._custom_ops['custom_profiler_decorator'] = self._custom_interface_manager.\
                get_interface('custom_profiler_decorator')
        self._custom_ops['custom_flash_attn'] = self._custom_interface_manager.\
                get_interface('custom_flash_attn')(flashattn_func_npu, rearrange_npu)

    def register_ops(self, args):
        """register_ops"""
        from .npu_ops import load
        load(args)

        # Register post-init custom ops
        try:
            import parallel_attention
        except ImportError:
            raise RuntimeError(
                "Failed to import post-init custom ops."
            )

        self._custom_ops['custom_parallel_attention'] = self._custom_interface_manager.\
                get_interface('custom_parallel_attention')(parallel_attention)

    def get_operator(self, op_name):
        """get_operator"""
        return self._custom_ops.get(op_name, None)
