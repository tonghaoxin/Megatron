# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""CUDACustomOperatorManager."""

import os
import pkgutil
import importlib
import torch

from .abstract_custom_operator_manager import MegatronCustomOperatorManager
from ..interfaces import CUDAInterfaceManager


class CUDACustomOperatorManager(MegatronCustomOperatorManager):
    """CUDACustomOperatorManager"""

    def __init__(self):
        """__init__"""
        self._custom_ops = {}
        self._custom_interface_manager = CUDAInterfaceManager()

        # Register pre-init custom ops
        try:
            import fused_weight_gradient_mlp_cuda
            from torch import bmm as bmm_cuda
            from torch import baddbmm as baddbmm_cuda
            from nvtx import annotate as profiler_decorator_cuda
            from apex.parallel import SyncBatchNorm as sync_batch_norm_cuda
            from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction as fused_layer_norm_cuda
            from apex.normalization.fused_layer_norm import FusedRMSNorm as fused_rmsnorm_cuda
            from apex.contrib.layer_norm.layer_norm import FastLayerNormFN as fast_layer_norm_cuda
            from apex.multi_tensor_apply import multi_tensor_applier as multi_tensor_applier_cuda
            from amp_C import multi_tensor_l2norm as multi_tensor_l2norm_cuda
            from amp_C import multi_tensor_scale as multi_tensor_scale_cuda
            # from flash_attn.flash_attn_interface import flash_attn_unpadded_func as flashattn_func_cuda # pip install flash-attn
            from flash_attn.flash_attn_interface import flash_attn_varlen_func as flashattn_func_cuda
            from einops import rearrange as rearrange_cuda # pip install einops'
        except ImportError:
            raise RuntimeError(
                "Failed to import pre-init custom ops."
            )

        self._custom_ops['custom_bmm'] = bmm_cuda
        self._custom_ops['custom_baddbmm'] = baddbmm_cuda
        self._custom_ops['custom_profiler_decorator'] = profiler_decorator_cuda
        self._custom_ops['custom_sync_batch_norm'] = sync_batch_norm_cuda
        self._custom_ops['fused_layer_norm'] = fused_layer_norm_cuda
        self._custom_ops['fast_layer_norm'] = fast_layer_norm_cuda
        self._custom_ops['custom_rms_norm'] = self._custom_interface_manager.\
                 get_interface('custom_rms_norm').wrap(fused_rmsnorm_cuda)
        self._custom_ops['custom_fc'] = self._custom_interface_manager.\
                get_interface('custom_fc')()
        self._custom_ops['custom_grad_accum'] = self._custom_interface_manager.\
                get_interface('custom_grad_accum')(fused_weight_gradient_mlp_cuda)
        self._custom_ops['custom_multi_tensor_applier'] = self._custom_interface_manager.\
                get_interface('custom_multi_tensor_applier')(
            multi_tensor_applier_cuda,
            {'l2norm': multi_tensor_l2norm_cuda, 'scale': multi_tensor_scale_cuda}
        )
        self._custom_ops['custom_flash_attn'] = self._custom_interface_manager.\
                get_interface('custom_flash_attn')(flashattn_func_cuda, rearrange_cuda)

    def _monkey_patch(self, args):
        """_monkey_patch"""
        pass

    def register_ops(self, args):
        """register_ops"""
        from .cuda_ops import load
        load(args)

        # Register post-init custom ops
        try:
            import scaled_upper_triang_masked_softmax_cuda
            import scaled_masked_softmax_cuda
            import scaled_softmax_cuda
            import rotary_positional_embedding_cuda
            import parallel_attention
            import matmul_reduce_parallel_cuda
            from megatron.core.parallel_state import get_tensor_model_parallel_group
        except ImportError:
            raise RuntimeError(
                "Failed to import post-init custom ops."
            )

        self._custom_interface_manager.set_inner_func(
            'scaled_upper_triang_masked_softmax',
            scaled_upper_triang_masked_softmax_cuda
        )

        self._custom_interface_manager.set_inner_func(
            'scaled_masked_softmax',
            scaled_masked_softmax_cuda
        )

        self._custom_interface_manager.set_inner_func(
            'scaled_softmax',
            scaled_softmax_cuda
        )

        self._custom_interface_manager.set_inner_func(
            'rotary_no_freqs_pos_emb',
            rotary_positional_embedding_cuda
        )

        self._custom_interface_manager.set_inner_func(
            'matmul_reduce_parallel',
            matmul_reduce_parallel_cuda
        )

        self._custom_interface_manager.set_inner_func(
            'tp_group_func',
            get_tensor_model_parallel_group
        )

        self._custom_ops['scaled_upper_triang_masked_softmax'] = self._custom_interface_manager.\
                get_interface('custom_scaled_upper_triang_masked_softmax')
        self._custom_ops['scaled_masked_softmax'] = self._custom_interface_manager.\
                get_interface('custom_scaled_masked_softmax')
        self._custom_ops['scaled_softmax'] = self._custom_interface_manager.\
                get_interface('custom_scaled_softmax')
        self._custom_ops['custom_apply_rope_with_freqs'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_with_freqs')
        self._custom_ops['custom_apply_rope_no_freqs'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_no_freqs')
        self._custom_ops['custom_apply_rope_with_idx'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_with_idx')
        self._custom_ops['custom_parallel_attention'] = self._custom_interface_manager.\
                get_interface('custom_parallel_attention')(parallel_attention)

        # Monkey patch custom classes and modules
        self._monkey_patch(args)

    def get_operator(self, op_name):
        """get_operator"""
        return self._custom_ops.get(op_name, None)
