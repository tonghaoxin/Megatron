# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""CUDAInterfaceManager."""

import os
import pkgutil
import importlib
import torch
from typing import Optional

from .abstract_interface_manager import MegatronInterfaceManager
from .abstract_interface_manager import _INTERFACE_INNER_FUNCS
from .abstract_interface_manager import _rotate_half_v1, _rotate_half_v2


def _apply_rope_with_freqs(q, k, freqs):
    """_apply_rope_with_freqs"""
    q = (q * freqs.cos()) + (_rotate_half_v1(q) * freqs.sin())
    k = (k * freqs.cos()) + (_rotate_half_v1(k) * freqs.sin())
    return q, k


def _apply_rope_with_idx(q, k, cos, sin, original=False):
    """_apply_rope_with_idx"""
    if original:
        q = q.reshape(q.size(0), -1, q.size(2), q.size(3) // 2, 2)
        q = torch.cat([q[..., 0], q[..., 1]], dim=-1)
        k = k.reshape(k.size(0), -1, k.size(2), k.size(3) // 2, 2)
        k = torch.cat([k[..., 0], k[..., 1]], dim=-1)
        q, k = (q * cos) + (_rotate_half_v2(q) * sin), (k * cos) + (_rotate_half_v2(k) * sin)
        s1, s2, s3, s4 = q.size(0), q.size(1), q.size(2), q.size(3)
        k1, k2, k3, k4 = k.size(0), k.size(1), k.size(2), k.size(3)
        q = torch.stack(torch.chunk(q, 2, -1), dim=-1)
        q = q.reshape(s1, s2, s3, s4)
        k = torch.stack(torch.chunk(k, 2, -1), dim=-1)
        k = k.reshape(k1, k2, k3, k4)
    else:
        q, k = (q * cos) + (_rotate_half_v2(q) * sin), (k * cos) + (_rotate_half_v2(k) * sin)
    return q, k


class CustomFCWrapper(object):
    """CustomFC wrapper"""

    def __init__(self):
        """__init__"""
        pass

    def __call__(self, *args, **kwargs):
        """__call__"""
        global _INTERFACE_INNER_FUNCS
        tp_gemm_func = _INTERFACE_INNER_FUNCS.get('matmul_reduce_parallel', None)
        tp_group_func = _INTERFACE_INNER_FUNCS.get('tp_group_func', None)
        ctx = args[0]
        total_input = args[1]
        weight = args[2]
        bias = args[3]
        prepared = kwargs.get('is_forward', False) and kwargs.get('is_row_parallel', False)
        kernel_split_num = kwargs.get('kernel_split_num', None)
        if tp_gemm_func is not None and \
                tp_group_func is not None and \
                kwargs.get('tp_gemm_parallel', False) and \
                prepared and kernel_split_num is not None:
            output = tp_gemm_func.matmul_reduce_parallel(
                total_input, weight,
                not kwargs.get('transpose', False),
                kernel_split_num,
                torch.distributed.get_rank(),
                tp_group_func())
        else:
            if kwargs.get('transpose', False):
                weight = weight.t()
            output = torch.matmul(total_input, weight)
        if bias is not None:
            output = output + bias
        return output


class CustomRMSNormWrapper(object):
    """CustomRMSNorm wrapper"""

    @staticmethod
    def wrap(parent_class):
        """wrap"""

        class CustomRMSNorm(parent_class):
            """Inherit from torch_xmlir.nn.rms_norm but add an invalid argument: weight_empty"""

            def __init__(self, normalized_shape, eps=1e-5,
                         elementwise_affine=True,
                         sequence_parallel=False,
                         weight_empty=False):
                """__init__"""
                super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

                if weight_empty:
                    self.weight = torch.nn.Parameter(torch.empty(normalized_shape))
                else:
                    self.weight = torch.nn.Parameter(torch.ones(normalized_shape))

                self.sequence_parallel = sequence_parallel
                # set sequence parallelism flag on weight parameters
                setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

        return CustomRMSNorm


class CustomGradAccumWrapper(object):
    """CustomGradAccum wrapper"""

    def __init__(self, fused_func):
        """__init__"""
        self.fused_func = fused_func

    def __call__(self, *args, **kwargs):
        """__call__"""
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


class CustomApplyRoPEWrapper(object):
    """CustomApplyRoPE wrapper"""

    def __init__(self, rope_func):
        """__init__"""
        self.rope_func = rope_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.rope_func(*args, **kwargs)

class CustomFlashAttnWrapper(object):
    """CustomFlashAttn wrapper"""

    def __init__(self, flashattn_func, rearrange):
        """__init__"""
        self.flashattn_func = flashattn_func
        self.rearrange = rearrange

    def __call__(self, *args, **kwargs):
        """__call__"""
        q = args[0]
        k = args[1]
        v = args[2]
        # attention_mask = args[3]
        nheads = args[4]
        is_training = args[5]
        is_causal = args[6]
        # scale = args[7]
        dropout_p = args[8]

        q, k, v = [self.rearrange(x, 's b ... -> b s ...').contiguous()
                   for x in (q, k, v)]

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [self.rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if is_training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q
            cu_seqlens_k = cu_seqlens_q
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = (seqlen_q == seqlen_k)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            dropout_p = 0

        output = self.flashattn_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p, softmax_scale=None, causal=is_causal
        )

        context_layer = self.rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        output = self.rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

        return output


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
    def get_batch_per_block(sq, sk, b, np):
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
    def forward(ctx, q, k, cos, sin):
        """forward"""
        global _INTERFACE_INNER_FUNCS
        q_ = q.contiguous()
        k_ = k.contiguous()
        cos_ = cos.contiguous()
        sin_ = sin.contiguous()
        output_q = _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb', None).forward(
            q_, cos_, sin_, False
        )
        output_k = _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb', None).forward(
            k_, cos_, sin_, False
        )
        ctx.save_for_backward(cos_, sin_)

        return output_q, output_k

    @staticmethod
    def backward(ctx, dq_emb, dk_emb):
        """backward"""
        global _INTERFACE_INNER_FUNCS
        cos_, sin_ = ctx.saved_tensors
        grad_q = _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb', None).backward(
            dq_emb, cos_, sin_, False
        )
        grad_k = _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb', None).backward(
            dk_emb, cos_, sin_, False
        )

        return grad_q, grad_k, None, None


class CUDAInterfaceManager(MegatronInterfaceManager):
    """CUDAInterfaceManager"""

    def __init__(self):
        """__init__"""
        self._custom_interfaces = {}

        self._custom_interfaces['custom_fc'] = CustomFCWrapper
        self._custom_interfaces['custom_grad_accum'] = CustomGradAccumWrapper
        self._custom_interfaces['custom_multi_tensor_applier'] = CustomMultiTensorApplyWrapper
        self._custom_interfaces['custom_parallel_attention'] = CustomParallelAttentionWrapper
        self._custom_interfaces['custom_flash_attn'] = CustomFlashAttnWrapper
        self._custom_interfaces['custom_scaled_upper_triang_masked_softmax'] = \
                CustomSoftmaxWrapper(ScaledUpperTriangMaskedSoftmax)
        self._custom_interfaces['custom_scaled_masked_softmax'] = CustomSoftmaxWrapper(ScaledMaskedSoftmax)
        self._custom_interfaces['custom_scaled_softmax'] = CustomSoftmaxWrapper(ScaledSoftmax)
        self._custom_interfaces['custom_apply_rope_with_freqs'] = \
                CustomApplyRoPEWrapper(_apply_rope_with_freqs)
        self._custom_interfaces['custom_apply_rope_no_freqs'] = \
                CustomApplyRoPEWrapper(RotaryPositionalEmbeddingFunction.apply)
        self._custom_interfaces['custom_apply_rope_with_idx'] = \
                CustomApplyRoPEWrapper(_apply_rope_with_idx)
        self._custom_interfaces['custom_rms_norm'] = CustomRMSNormWrapper

    def set_inner_func(self, interface_name, inner_func):
        """set_inner_func"""
        global _INTERFACE_INNER_FUNCS
        _INTERFACE_INNER_FUNCS[interface_name] = inner_func

    def get_interface(self, interface_name):
        """get_interface"""
        return self._custom_interfaces[interface_name]
