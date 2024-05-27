# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""XPUInterfaceManager."""

import inspect
import torch
import os
from typing import Optional

from functools import wraps
from .abstract_interface_manager import MegatronInterfaceManager
from .abstract_interface_manager import _INTERFACE_INNER_FUNCS
from .abstract_interface_manager import _rotate_half_v2


def _apply_rope_with_idx(q, k, cos, sin, original=False):
    """_apply_rope_with_idx"""
    if original:
        q, k = _INTERFACE_INNER_FUNCS.get('rotary_with_idx_pos_emb', None)(q, k, cos, sin, "LBHD")
    else:
        q, k = (q * cos) + (_rotate_half_v2(q) * sin), (k * cos) + (_rotate_half_v2(k) * sin)
    return q, k


def bias_dropout_add(x, bias, residual, prob, training):
    """bias_dropout_add"""
    if bias is not None:
        x = x + bias

    if abs(prob) < 1e-5:
        out = residual + x
        return out

    out = _INTERFACE_INNER_FUNCS.get('dropout_add')(x.view(x.shape[0], -1), \
            residual.view(x.shape[0], -1), prob, not training)
    return out.view(x.shape)


def get_bias_dropout_add(training):
    """get_bias_dropout_add"""
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    """bias_dropout_add_fused_train"""
    return bias_dropout_add(x, bias, residual, prob, True)


def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: Optional[torch.Tensor],
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    """bias_dropout_add_fused_inference"""
    return bias_dropout_add(x, bias, residual, prob, False)


class CustomLayerNormWrapper(object):
    """CustomLayerNorm wrapper"""

    @staticmethod
    def wrap(parent_class):
        """wrap"""

        class CustomLayerNorm(parent_class):
            """Inherit from torch.nn.LayerNorm but eliminate extra kwargs"""

            def __init__(self, normalized_shape, eps=1e-5,
                        no_persist_layer_norm=True,
                        sequence_parallel=False,
                        apply_layernorm_1p=False):
                    super().__init__(normalized_shape, eps = eps)
                    self.sequence_parallel = sequence_parallel
                    setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
                    setattr(self.bias, 'sequence_parallel', self.sequence_parallel)

        return CustomLayerNorm


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
                super().__init__(normalized_shape, eps=eps, sequence_parallel=sequence_parallel)

                if weight_empty:
                    self.weight = torch.nn.Parameter(torch.empty(normalized_shape))
                else:
                    self.weight = torch.nn.Parameter(torch.ones(normalized_shape))

                self.sequence_parallel = sequence_parallel
                # set sequence parallelism flag on weight parameters
                setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

        return CustomRMSNorm


class CustomFCWrapper(object):
    """CustomFC wrapper"""

    def __init__(self, findmax, fc_fusion, linear):
        """__init__"""
        self.findmax = findmax
        self.fc_fusion = fc_fusion
        self.linear = linear
        self.gemm_type = os.getenv("XDNN_FC_GEMM_DTYPE")

    def __call__(self, *args, **kwargs):
        """__call__"""
        ctx = args[0]
        total_input = args[1]
        weight = args[2]
        bias = args[3]
        if kwargs.get('findmax_opt', False):
            device = total_input.device
            if kwargs.get('is_forward', False):
                if self.gemm_type == "float32":
                    input_max, weight_max = None, None
                else:
                    input_max = torch.empty((64), dtype=torch.float, device=device)
                    self.findmax(total_input, max=input_max)
                    weight_max = torch.empty((64), dtype=torch.float, device=device)
                    self.findmax(weight, max=weight_max)
                ctx.input_max = input_max
                ctx.weight_max = weight_max
                output = total_input.new_empty(total_input.shape[0], total_input.shape[1], weight.shape[0])
                self.fc_fusion(
                    total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[-1]),
                    weight,
                    other_trans=True,
                    self_max=input_max,
                    other_max=weight_max,
                    out=output
                )
            else: # Backward
                if self.gemm_type == "float32":
                    grad_output_max = None
                else:
                    grad_output_max = torch.empty((64), dtype=torch.float, device=device)
                    self.findmax(total_input, max=grad_output_max)
                ctx.grad_output_max = grad_output_max
                output = total_input.new_empty(total_input.shape[:-1] + (weight.shape[-1],))
                # dims compression for mat1 and mat2 dims not same
                tmp_grad_output = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[-1]) #ok
                self.fc_fusion(
                    tmp_grad_output,
                    weight,
                    self_max=grad_output_max,
                    other_max=ctx.weight_max,
                    out=output
                )
        else:
            if kwargs.get('is_forward', False):
                output = self.linear(total_input, weight, bias)
            else: # Backward
                output = total_input.matmul(weight)
        return output


class CustomGradAccumWrapper(object):
    """CustomGradAccum wrapper"""

    def __init__(self, fused_func):
        """__init__"""
        self.fused_func = fused_func

    def __call__(self, *args, **kwargs):
        """__call__"""
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


class CustomFlashAttnWrapper(object):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
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
        softmax_scale = args[7]
        dropout_p = args[8]


        q, k, v = [self.rearrange(x, 's b ... -> b s ...').contiguous()
                   for x in (q, k, v)]
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [self.rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        # cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
        #                             device=q.device)
        # WARN(wangfangfei): moved the tensor from cuda to cpu
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32)

        if is_training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            # is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            dropout_p = 0

        output = self.flashattn_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p,
            softmax_scale=softmax_scale, causal=is_causal
        )

        context_layer = self.rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        output = self.rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
        return output

    def extra_repr(self):
        """extra_repr"""
        return 'causal={}, softmax_scale={}, attention_dropout={}'.format(
                self.causal, self.softmax_scale, self.dropout_p)


class CustomMultiTensorApplyWrapper(object):
    """CustomMultiTensorApply wrapper"""

    def __init__(self, multi_tensor_applier, multi_tensor_ops):
        """__init__"""
        self.multi_tensor_applier = multi_tensor_applier
        self.multi_tensor_ops = multi_tensor_ops

    def __call__(self, *args, **kwargs):
        """__call__"""
        if kwargs.get('multi_tensor_fusion', True):
            return self.multi_tensor_applier(self.multi_tensor_ops[kwargs.get('op', 'scale')], *args)
        else:
            grads_for_norm = args[1][0]
            grad_norm = None
            if kwargs.get('op', 'scale') == 'l2norm':
                grad_norm = sum([item.square().sum() for item in grads_for_norm]).sqrt()
                return grad_norm, None
            elif kwargs.get('op', 'scale') == 'scale':
                clip_coeff = args[2]
                [grad.mul_(clip_coeff) for grad in grads_for_norm]
                return
            

class CustomParallelAttentionWrapper(object):
    """CustomParallelAttention wrapper"""

    def __init__(self, parallel_attention_func):
        """__init__"""
        self.parallel_attention_func = parallel_attention_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.parallel_attention_func.update(*args)


class CustomProfilerDecoratorWrapper(object):
    """CustomProfilerDecorator wrapper"""

    def __init__(self, message=None, color=None, domain=None, category=None):
        """__init__"""
        pass

    def __call__(self, func):
        """__call__"""
        @wraps(func)
        def inner(*args, **kwargs):
            """inner"""
            return func(*args, **kwargs)

        return inner

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



class CustomSwiGLUWrapper(object):
    """CustomSwiGLUWrapper wrapper"""

    def __init__(self, swiglu):
        """__init__"""
        if inspect.isclass(swiglu):
            self.swiglu = swiglu()
        elif callable(swiglu):
            self.swiglu = swiglu

    def __call__(self, *args, **kwargs):
        """__call__"""
        x = args[0]
        return self.swiglu(x)


class CustomApplyRoPEWrapper(object):
    """CustomApplyRoPE wrapper"""

    def __init__(self, rope_func):
        """__init__"""
        self.rope_func = rope_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.rope_func(*args, **kwargs)

class CustomDropoutAddWrapper(object):
    """CustomDropoutAdd wrapper"""

    def __init__(self, dropout_add_func):
        """__init__"""
        self.dropout_add_func = dropout_add_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.dropout_add_func(*args, **kwargs)


class RotaryPositionalEmbeddingFunction(torch.autograd.Function):
    """RotaryPositionalEmbeddingFunction"""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        """forward"""
        global _INTERFACE_INNER_FUNCS
        output_q = q.new_empty(q.shape)
        output_k = k.new_empty(k.shape)
        _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb_forward', None)(q, cos, sin, out=output_q)
        _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb_forward', None)(k, cos, sin, out=output_k)
        ctx.save_for_backward(cos, sin)

        return output_q, output_k

    @staticmethod
    def backward(ctx, dq_emb, dk_emb):
        """backward"""
        global _INTERFACE_INNER_FUNCS
        cos_, sin_ = ctx.saved_tensors
        grad_q = dq_emb.new_empty(dq_emb.shape)
        grad_k = dk_emb.new_empty(dk_emb.shape)
        _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb_backward', None)(dq_emb, cos_, sin_, out=grad_q)
        _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb_backward', None)(dk_emb, cos_, sin_, out=grad_k)

        return grad_q, grad_k, None, None


class RotaryPositionalEmbeddingWithFreqsFunction(torch.autograd.Function):
    """RotaryPositionalEmbeddingWithFreqsFunction"""

    @staticmethod
    def forward(ctx, q, k, freqs):
        """forward"""
        global _INTERFACE_INNER_FUNCS
        q_emb = q.new_empty(q.shape)
        k_emb = k.new_empty(k.shape)
        _INTERFACE_INNER_FUNCS.get('rotary_with_freqs_pos_emb_qk_forward', None)(q, k, freqs, q_emb=q_emb, k_emb=k_emb)
        ctx.freqs = freqs
        return q_emb, k_emb

    @staticmethod
    def backward(ctx, dq_emb, dk_emb):
        """backward"""
        global _INTERFACE_INNER_FUNCS
        freqs_ = ctx.freqs
        dq = dq_emb.new_empty(dq_emb.shape)
        dk = dk_emb.new_empty(dk_emb.shape)
        _INTERFACE_INNER_FUNCS.get('rotary_with_freqs_pos_emb_qk_backward', None)(dq_emb, dk_emb, freqs_, dq=dq, dk=dk)
        return dq, dk, None


class XPUInterfaceManager(MegatronInterfaceManager):
    """XPUInterfaceManager"""

    def __init__(self):
        """__init__"""
        global _INTERFACE_INNER_FUNCS
        self._custom_interfaces = {}

        self._custom_interfaces['custom_layer_norm'] = CustomLayerNormWrapper
        self._custom_interfaces['custom_rms_norm'] = CustomRMSNormWrapper
        self._custom_interfaces['custom_fc'] = CustomFCWrapper
        self._custom_interfaces['custom_grad_accum'] = CustomGradAccumWrapper
        self._custom_interfaces['custom_multi_tensor_applier'] = CustomMultiTensorApplyWrapper
        self._custom_interfaces['custom_parallel_attention'] = CustomParallelAttentionWrapper
        self._custom_interfaces['custom_flash_attn'] = CustomFlashAttnWrapper
        self._custom_interfaces['custom_profiler_decorator'] = CustomProfilerDecoratorWrapper
        self._custom_interfaces['custom_swiglu'] = CustomSwiGLUWrapper
        self._custom_interfaces['custom_apply_rope_with_freqs'] = \
                CustomApplyRoPEWrapper(RotaryPositionalEmbeddingWithFreqsFunction.apply)
        self._custom_interfaces['custom_apply_rope_no_freqs'] = \
                CustomApplyRoPEWrapper(RotaryPositionalEmbeddingFunction.apply)
        self._custom_interfaces['custom_apply_rope_with_idx'] = \
                CustomApplyRoPEWrapper(_apply_rope_with_idx)
        self._custom_interfaces['custom_scaled_upper_triang_masked_softmax'] = \
                CustomSoftmaxWrapper(ScaledUpperTriangMaskedSoftmax)
        self._custom_interfaces['custom_scaled_masked_softmax'] = CustomSoftmaxWrapper(ScaledMaskedSoftmax)
        self._custom_interfaces['custom_scaled_softmax'] = CustomSoftmaxWrapper(ScaledSoftmax)
        self._custom_interfaces['custom_bias_dropout_add_fused_train'] = \
                CustomDropoutAddWrapper(bias_dropout_add_fused_train)
        self._custom_interfaces['custom_bias_dropout_add_fused_inference'] = \
                CustomDropoutAddWrapper(bias_dropout_add_fused_inference)

    def set_inner_func(self, interface_name, inner_func):
        """set_inner_func"""
        global _INTERFACE_INNER_FUNCS
        _INTERFACE_INNER_FUNCS[interface_name] = inner_func

    def get_interface(self, interface_name):
        """get_interface"""
        return self._custom_interfaces[interface_name]
