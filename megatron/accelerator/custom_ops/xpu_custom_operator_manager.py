# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""XPUCustomOperatorManager."""

import os
import pkgutil
import importlib
import torch

from .abstract_custom_operator_manager import MegatronCustomOperatorManager
from ..interfaces import XPUInterfaceManager


class XPUCustomOperatorManager(MegatronCustomOperatorManager):
    """XPUAccelerator"""

    def __init__(self):
        """__init__"""
        self._custom_ops = {}
        self._custom_interface_manager = XPUInterfaceManager()

        # Register pre-init custom ops
        try:
            # import scaled_upper_triang_masked_softmax_cuda
            # import scaled_masked_softmax_cuda
            # import scaled_softmax_cuda
            import fused_weight_gradient_mlp_cuda
            from torch.nn import LayerNorm
            from torch_xmlir.nn.rms_norm import RMSNormLayer
            from torch_xmlir.nn.bmm import bmm as bmm_xpu
            from torch_xmlir.nn.baddbmm import baddbmm as baddbmm_xpu
            from torch_xmlir.nn.swiglu import SwiGLU as swiglu_xpu
            from torch_xmlir.nn.linear import linear as linear_xpu
            from torch_xmlir.nn.rotary_pos_emb_index import RotaryPosEmbIdx
            from apex.multi_tensor_apply import multi_tensor_applier as multi_tensor_applier_xpu
            from amp_C import multi_tensor_l2norm as multi_tensor_l2norm_xpu
            from amp_C import multi_tensor_scale as multi_tensor_scale_xpu
            from flash_attn.flash_attn_interface import flash_attn_varlen_func as flashattn_func_xpu
            from einops import rearrange as rearrange_xpu # pip install einops'
            from torch_xmlir.nn.dropout_add import DropoutAddFunction
            # TODO(changtao02): try to check them they are importable
            findmax_xpu = torch.ops.custom_ops.findmax
            fc_fusion_xpu = torch.ops.custom_ops.fc_fusion
            rotary_with_freqs_pos_emb_forward_xpu = torch.ops.custom_ops.rotary_pos_emb
            rotary_with_freqs_pos_emb_backward_xpu = torch.ops.custom_ops.rotary_pos_emb_backward
            rotary_no_freqs_pos_emb_forward_xpu = torch.ops.custom_ops.rotary_no_freqs_pos_emb_forward
            rotary_no_freqs_pos_emb_backward_xpu = torch.ops.custom_ops.rotary_no_freqs_pos_emb_backward
            rotary_with_freqs_pos_emb_qk_forward_xpu = torch.ops.custom_ops.rotary_pos_emb_qk
            rotary_with_freqs_pos_emb_qk_backward_xpu = torch.ops.custom_ops.rotary_pos_emb_qk_backward
            rotary_with_idx_pos_emb_xpu = RotaryPosEmbIdx()
            dropout_add_xpu = DropoutAddFunction.apply
        except ImportError:
            raise RuntimeError(
                "Failed to import pre-init custom ops."
            )

        # self._custom_interface_manager.set_inner_func(
        #     'scaled_upper_triang_masked_softmax',
        #     scaled_upper_triang_masked_softmax_cuda
        # )

        # self._custom_interface_manager.set_inner_func(
        #     'scaled_masked_softmax',
        #     scaled_masked_softmax_cuda
        # )

        # self._custom_interface_manager.set_inner_func(
        #     'scaled_softmax',
        #     scaled_softmax_cuda
        # )

        self._custom_interface_manager.set_inner_func(
            'rotary_with_freqs_pos_emb_forward',
            rotary_with_freqs_pos_emb_forward_xpu
        )

        self._custom_interface_manager.set_inner_func(
            'rotary_with_freqs_pos_emb_backward',
            rotary_with_freqs_pos_emb_backward_xpu
        )

        self._custom_interface_manager.set_inner_func(
            'rotary_no_freqs_pos_emb_forward',
            rotary_no_freqs_pos_emb_forward_xpu
        )

        self._custom_interface_manager.set_inner_func(
            'rotary_with_freqs_pos_emb_qk_forward',
            rotary_with_freqs_pos_emb_qk_forward_xpu
        )

        self._custom_interface_manager.set_inner_func(
            'rotary_with_freqs_pos_emb_qk_backward',
            rotary_with_freqs_pos_emb_qk_backward_xpu
        )

        self._custom_interface_manager.set_inner_func(
            'rotary_no_freqs_pos_emb_backward',
            rotary_no_freqs_pos_emb_backward_xpu
        )

        self._custom_interface_manager.set_inner_func(
            'rotary_with_idx_pos_emb',
            rotary_with_idx_pos_emb_xpu
        )

        self._custom_interface_manager.set_inner_func(
            'dropout_add',
            dropout_add_xpu
        )

        self._custom_ops['custom_layer_norm'] = self._custom_interface_manager.\
                get_interface('custom_layer_norm').wrap(LayerNorm)
        self._custom_ops['custom_rms_norm'] = self._custom_interface_manager.\
                get_interface('custom_rms_norm').wrap(RMSNormLayer)
        self._custom_ops['custom_bmm'] = bmm_xpu
        self._custom_ops['custom_baddbmm'] = baddbmm_xpu
        self._custom_ops['custom_multi_tensor_applier'] = self._custom_interface_manager.\
                get_interface('custom_multi_tensor_applier')(multi_tensor_applier_xpu, \
                    {'l2norm': multi_tensor_l2norm_xpu, 'scale': multi_tensor_scale_xpu})
        # self._custom_ops['scaled_upper_triang_masked_softmax'] = self._custom_interface_manager.\
        #         get_interface('custom_scaled_upper_triang_masked_softmax')
        # self._custom_ops['scaled_masked_softmax'] = self._custom_interface_manager.\
        #         get_interface('custom_scaled_masked_softmax')
        # self._custom_ops['scaled_softmax'] = self._custom_interface_manager.\
        #         get_interface('custom_scaled_softmax')
        self._custom_ops['custom_swiglu'] = self._custom_interface_manager.\
                get_interface('custom_swiglu')(swiglu_xpu)
        self._custom_ops['custom_fc'] = self._custom_interface_manager.\
                get_interface('custom_fc')(findmax_xpu, fc_fusion_xpu, linear_xpu)
        self._custom_ops['custom_grad_accum'] = self._custom_interface_manager.\
                get_interface('custom_grad_accum')(fused_weight_gradient_mlp_cuda)
        self._custom_ops['custom_apply_rope_with_freqs'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_with_freqs')
        self._custom_ops['custom_apply_rope_no_freqs'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_no_freqs')
        self._custom_ops['custom_apply_rope_with_idx'] = self._custom_interface_manager.\
                get_interface('custom_apply_rope_with_idx')
        self._custom_ops['custom_profiler_decorator'] = self._custom_interface_manager.\
                get_interface('custom_profiler_decorator')
        self._custom_ops['sync_batch_norm'] = None # TODO(changtao02): add xpu custom SyncBatchNorm
        self._custom_ops['custom_flash_attn'] = self._custom_interface_manager.\
                get_interface('custom_flash_attn')(flashattn_func_xpu, rearrange_xpu)
        self._custom_ops['custom_bias_dropout_add_fused_train'] = self._custom_interface_manager.\
                get_interface('custom_bias_dropout_add_fused_train')
        self._custom_ops['custom_bias_dropout_add_fused_inference'] = self._custom_interface_manager.\
                get_interface('custom_bias_dropout_add_fused_inference')

    def _monkey_patch(self, args):
        """_monkey_patch"""
        pass

    def register_ops(self, args):
        """register_ops"""
        from .xpu_ops import load
        load(args)

        # Register post-init custom ops
        try:
            import parallel_attention
        except ImportError:
            raise RuntimeError(
                "Failed to post-init custom ops."
            )

        self._custom_ops['custom_parallel_attention'] = self._custom_interface_manager.\
                get_interface('custom_parallel_attention')(parallel_attention)

        # Monkey patch custom classes and modules
        self._monkey_patch(args)

    def get_operator(self, op_name):
        """get_operator"""
        return self._custom_ops.get(op_name, None)
