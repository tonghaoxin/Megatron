# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""CUDAInterfaceManager."""

import os
import pkgutil
import importlib
import torch

from .abstract_interface_manager import MegatronInterfaceManager
from .abstract_interface_manager import _INTERFACE_INNER_FUNCS


class CustomFCWrapper(object):
    """CustomFC wrapper"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        ctx = args[0]
        total_input = args[1]
        weight = args[2]
        bias = args[3]
        if kwargs.get('transpose', False):
            weight = weight.t()
        output = torch.matmul(total_input, weight)
        if bias is not None:
            output = output + bias
        return output


class CustomGradAccumWrapper(object):
    """CustomGradAccum wrapper"""

    def __init__(self, fused_func):
        self.fused_func = fused_func

    def __call__(self, *args, **kwargs):
        # ctx = args[0]
        total_input = args[1]
        grad_output = args[2]
        weight_grad = args[3]
        if kwargs.get('to_fuse', False):
            if weight_grad.dtype == torch.float32:
                self.fused_func.wgrad_gemm_accum_fp32(total_input, grad_output, weight_grad)
            elif weight_grad.dtype == torch.float16:
                self.fused_func.wgrad_gemm_accum_fp16(total_input, grad_output, weight_grad)
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        return grad_weight


class CustomMultiTensorApplyWrapper(object):
    """CustomMultiTensorApply wrapper"""

    def __init__(self, multi_tensor_applier, multi_tensor_ops):
        """__init__"""
        self.multi_tensor_applier = multi_tensor_applier
        self.multi_tensor_ops = multi_tensor_ops

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.multi_tensor_applier(self.multi_tensor_ops[kwargs.get('op', 'scale')], *args)


class CustomParallelAttentionWrapper(object):
    """CustomParallelAttention wrapper"""

    def __init__(self, parallel_attention_func):
        """__init__"""
        self.parallel_attention_func = parallel_attention_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.parallel_attention_func.update(*args)


class CustomSoftmaxWrapper(object):
    """CustomSoftmax wrapper"""

    def __init__(self, autograd_func):
        """__init__"""
        self.autograd_func = autograd_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.autograd_func.apply(*args)

    def get_batch_per_block(self, sq, sk, b, np):
        """get_batch_per_block"""
        return self.autograd_func.get_batch_per_block(sq, sk, b, np)


class CustomRoPEWrapper(object):
    """CustomRoPE wrapper"""

    def __init__(self, rope_func):
        """__init__"""
        self.rope_func = rope_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.rope_func.apply(*args)


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        """forward"""
        global _INTERFACE_INNER_FUNCS
        scale_t = torch.tensor([scale])
        softmax_results = _INTERFACE_INNER_FUNCS.get('scaled_upper_triang_masked_softmax', None).forward(
            inputs, scale_t[0]
        )

        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        """backward"""
        global _INTERFACE_INNER_FUNCS
        softmax_results, scale_t = ctx.saved_tensors
        input_grads = _INTERFACE_INNER_FUNCS.get('scaled_upper_triang_masked_softmax', None).backward(
            output_grads, softmax_results, scale_t[0]
        )

        return input_grads, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the mask.
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        """forward"""
        global _INTERFACE_INNER_FUNCS
        scale_t = torch.tensor([scale])

        softmax_results = _INTERFACE_INNER_FUNCS.get('scaled_masked_softmax', None).forward(
            inputs, mask, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        """backward"""
        global _INTERFACE_INNER_FUNCS
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = _INTERFACE_INNER_FUNCS.get('scaled_masked_softmax', None).backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None

    @staticmethod
    def get_batch_per_block(ctx, sq, sk, b, np):
        """get_batch_per_block"""
        global _INTERFACE_INNER_FUNCS
        return _INTERFACE_INNER_FUNCS.get('scaled_masked_softmax', None).get_batch_per_block(sq, sk, b, np)


class ScaledSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following two operations in sequence
    1. Scale the tensor.
    2. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        """forward"""
        global _INTERFACE_INNER_FUNCS
        scale_t = torch.tensor([scale])

        softmax_results = _INTERFACE_INNER_FUNCS.get('scaled_softmax', None).forward(
            inputs, scale_t[0]
        )
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        """backward"""
        global _INTERFACE_INNER_FUNCS
        softmax_results, scale_t = ctx.saved_tensors

        input_grads = _INTERFACE_INNER_FUNCS.get('scaled_softmax', None).backward(
            output_grads, softmax_results, scale_t[0]
        )
        return input_grads, None, None


class RotaryPositionalEmbeddingFunction(torch.autograd.Function):
    """CustomRotaryPositionalEmbeddingFunction"""

    @staticmethod
    def forward(ctx, q, cos, sin):
        """forward"""
        global _INTERFACE_INNER_FUNCS
        q_ = q.contiguous()
        cos_ = cos.contiguous()
        sin_ = sin.contiguous()
        output = _INTERFACE_INNER_FUNCS.get('rotary_positional_embedding', None).forward(
            *q.shape, q_, cos_, sin_
        )
        ctx.save_for_backward(cos_, sin_)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        global _INTERFACE_INNER_FUNCS
        cos_, sin_ = ctx.saved_tensors
        grad_q = _INTERFACE_INNER_FUNCS.get('rotary_positional_embedding', None).backward(
            *grad_output.shape, grad_output, cos_, sin_
        )

        return grad_q, None, None


class CUDAInterfaceManager(MegatronInterfaceManager):
    """CUDAInterfaceManager"""

    def __init__(self):
        """__init__"""
        self._custom_interfaces = {}

        self._custom_interfaces['custom_fc'] = CustomFCWrapper
        self._custom_interfaces['custom_grad_accum'] = CustomGradAccumWrapper
        self._custom_interfaces['custom_multi_tensor_applier'] = CustomMultiTensorApplyWrapper
        self._custom_interfaces['custom_parallel_attention'] = CustomParallelAttentionWrapper
        self._custom_interfaces['custom_scaled_upper_triang_masked_softmax'] = \
                CustomSoftmaxWrapper(ScaledUpperTriangMaskedSoftmax)
        self._custom_interfaces['custom_scaled_masked_softmax'] = CustomSoftmaxWrapper(ScaledMaskedSoftmax)
        self._custom_interfaces['custom_scaled_softmax'] = CustomSoftmaxWrapper(ScaledSoftmax)
        self._custom_interfaces['custom_rope'] = CustomRoPEWrapper(RotaryPositionalEmbeddingFunction)

    def set_inner_func(self, interface_name, inner_func):
        """set_inner_func"""
        global _INTERFACE_INNER_FUNCS
        _INTERFACE_INNER_FUNCS[interface_name] = inner_func

    def get_interface(self, interface_name):
        """get_interface"""
        return self._custom_interfaces[interface_name]
