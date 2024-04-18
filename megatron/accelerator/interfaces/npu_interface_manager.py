# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""NPUInterfaceManager."""

import torch
import inspect
from typing import Optional

from functools import wraps
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
    def wrap(parent_class, rms_norm_func):
        """wrap"""

        class CustomRMSNorm(parent_class):
            """Inherit from torch_xmlir.nn.rms_norm but add an invalid argument: weight_empty"""

            def __init__(self, normalized_shape, eps=1e-5,
                         elementwise_affine=True,
                         sequence_parallel=False,
                         weight_empty=False):
                """__init__"""
                super().__init__()
                self.eps = eps
                if weight_empty:
                    self.weight = torch.nn.Parameter(torch.empty(normalized_shape))
                else:
                    self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
                self.rms_norm_func = rms_norm_func

                self.sequence_parallel = sequence_parallel
                # set sequence parallelism flag on weight parameters
                setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

            def forward(self, x):
                """forward"""
                return self.rms_norm_func(x, self.weight, epsilon=self.eps)[0]

        return CustomRMSNorm


class CustomFCWrapper(object):
    """CustomFC wrapper"""

    def __init__(self):
        """__init__"""
        pass

    def __call__(self, *args, **kwargs):
        """__call__"""
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

    def __init__(self):
        """__init__"""
        pass

    def __call__(self, *args, **kwargs):
        """__call__"""
        # ctx = args[0]
        total_input = args[1]
        grad_output = args[2]
        weight_grad = args[3]
        grad_weight = grad_output.t().matmul(total_input)
        return grad_weight


class CustomMultiTensorApplyWrapper(object):
    """CustomMultiTensorApply wrapper"""

    def __init__(self):
        """__init__"""
        pass

    def __call__(self, *args, **kwargs):
        """__call__"""
        grads_for_norm = args[1][0]
        grad_norm = None
        if kwargs.get('op', 'scale') == 'l2norm':
            if not kwargs.get('memory_saving', False):
                grad_norm = 0.0
                norm_type = 2.0
                for item in grads_for_norm:
                    grad_norm += torch.norm(item, norm_type) ** norm_type
                grad_norm = grad_norm ** (1 / norm_type)
                # grad_norm = torch.cat([item.reshape(1, -1) for item in grads_for_norm], 1).to('cuda').norm()
            else:
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

    def __init__(self, custom_softmax_func):
        """__init__"""
        self.custom_softmax_func = custom_softmax_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        inputs = args[0]
        mask = args[1]
        scale = args[2]
        if scale is None:
            scale = 1.0

        return self.custom_softmax_func(inputs, mask, scale, False)

    def is_kernel_available(self, *args, **kwargs):
        """is_kernel_available"""
        mask = args[0]
        b = args[1]
        np = args[2]
        sq = args[3]
        sk = args[4]

        scaled_masked_softmax_fusion = kwargs.get('scaled_masked_softmax_fusion', False)
        input_in_float16 = kwargs.get('input_in_float16', False)

        return (
                scaled_masked_softmax_fusion  # user want to fuse
                and input_in_float16  # input must be fp16
                and 32 < sk <= 4096  # sk must be 32 ~ 4096
                and sq % 16 == 0  # sq must be divisor of 16
                and sk % 16 == 0  # sk must be divisor of 16
        )


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
        attention_mask = args[3]
        nheads = args[4]
        is_training = args[5]
        is_causal = args[6]
        scale = args[7]
        dropout_p = args[8]

        q, k, v = [self.rearrange(x, 's b h d -> s b (h d)').contiguous()
                   for x in (q, k, v)]

        seqlen_q, seqlen_k = q.shape[0], k.shape[0]

        if is_training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = (seqlen_q == seqlen_k)
            dropout_p = 0

        if is_causal:
            output = self.flashattn_func(
                q, k, v, nheads, "SBH",
                pse=None,
                padding_mask=None,
                atten_mask=attention_mask,
                scale=scale,
                pre_tockens=65536,  # seq_len
                next_tockens=0,  # 0
                keep_prob=1 - dropout_p,
                inner_precise=0)[0]
        else:
            raise Exception("Non-causal attention type is not support!")

        return output


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
        return self.swiglu(x, dim=-1)


class NPUInterfaceManager(MegatronInterfaceManager):
    """NPUInterfaceManager"""

    def __init__(self):
        """__init__"""
        self._custom_interfaces = {}

        self._custom_interfaces['custom_layer_norm'] = CustomLayerNormWrapper
        self._custom_interfaces['custom_rms_norm'] = CustomRMSNormWrapper
        self._custom_interfaces['custom_fc'] = CustomFCWrapper
        self._custom_interfaces['custom_grad_accum'] = CustomGradAccumWrapper
        self._custom_interfaces['custom_multi_tensor_applier'] = CustomMultiTensorApplyWrapper
        self._custom_interfaces['custom_parallel_attention'] = CustomParallelAttentionWrapper
        self._custom_interfaces['custom_profiler_decorator'] = CustomProfilerDecoratorWrapper
        self._custom_interfaces['custom_softmax'] = CustomSoftmaxWrapper
        self._custom_interfaces['custom_flash_attn'] = CustomFlashAttnWrapper
        self._custom_interfaces['custom_swiglu'] = CustomSwiGLUWrapper
        self._custom_interfaces['custom_apply_rope_no_freqs'] = CustomApplyRoPEWrapper
        self._custom_interfaces['custom_apply_rope_with_freqs'] = \
                CustomApplyRoPEWrapper(_apply_rope_with_freqs)
        self._custom_interfaces['custom_apply_rope_with_idx'] = \
                CustomApplyRoPEWrapper(_apply_rope_with_idx)

    def set_inner_func(self, interface_name, inner_func):
        """set_inner_func"""
        global _INTERFACE_INNER_FUNCS
        _INTERFACE_INNER_FUNCS[interface_name] = inner_func

    def get_interface(self, interface_name):
        """get_interface"""
        return self._custom_interfaces[interface_name]
