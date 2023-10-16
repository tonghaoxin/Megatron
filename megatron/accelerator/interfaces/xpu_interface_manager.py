# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""XPUInterfaceManager."""

import os
import pkgutil
import inspect
import importlib
import torch

from functools import wraps
from .abstract_interface_manager import MegatronInterfaceManager
from .abstract_interface_manager import _INTERFACE_INNER_FUNCS


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

        return CustomRMSNorm


class CustomFCWrapper(object):
    """CustomFC wrapper"""

    def __init__(self, findmax, fc_fusion, linear):
        """__init__"""
        self.findmax = findmax
        self.fc_fusion = fc_fusion
        self.linear = linear

    def __call__(self, *args, **kwargs):
        """__call__"""
        ctx = args[0]
        total_input = args[1]
        weight = args[2]
        bias = args[3]
        if kwargs.get('findmax_opt', False):
            device = total_input.device
            if kwargs.get('is_forward', False):
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

    def __init__(self, fc_fusion):
        """__init__"""
        self.fc_fusion = fc_fusion

    def __call__(self, *args, **kwargs):
        """__call__"""
        ctx = args[0]
        total_input = args[1]
        grad_output = args[2]
        # weight_grad = args[3]
        grad_weight = grad_output.new_empty(grad_output.shape[1], total_input.shape[1])
        if kwargs.get('findmax_opt', False):
            self.fc_fusion(grad_output, total_input, self_trans=True,
                           self_max=ctx.grad_output_max,
                           other_max=ctx.input_max,
                           out=grad_weight)
        else:
            self.fc_fusion(grad_output, total_input, self_trans=True, out=grad_weight)
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
                grad_norm = torch.cat([item.reshape(1, -1) for item in grads_for_norm], 1).to('xpu').norm()
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

    def __init__(self, autograd_func):
        """__init__"""
        self.autograd_func = autograd_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        inputs = args[0]
        mask = args[1]
        scale = args[2]

        mask = torch.cat([mask] * inputs.shape[1], 1)
        return self.autograd_func.apply(inputs, mask, -1, scale, -10000.0, True)


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


class CustomRoPEWrapper(object):
    """CustomRoPE wrapper"""

    def __init__(self, rope_func):
        """__init__"""
        self.rope_func = rope_func

    def __call__(self, *args, **kwargs):
        """__call__"""
        return self.rope_func.apply(*args)


class RotaryPositionalEmbeddingFunction(torch.autograd.Function):
    """RotaryPositionalEmbeddingFunction"""

    @staticmethod
    def forward(ctx, q, cos, sin):
        """forward"""
        global _INTERFACE_INNER_FUNCS
        output = q.new_empty(q.shape)
        _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb_forward', None)(q, cos, sin, out = output)
        ctx.save_for_backward(cos, sin)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """backward"""
        global _INTERFACE_INNER_FUNCS
        cos_, sin_ = ctx.saved_tensors
        grad_q = grad_output.new_empty(grad_output.shape)
        _INTERFACE_INNER_FUNCS.get('rotary_no_freqs_pos_emb_backward', None)(grad_output, cos_, sin_, out = grad_q)

        return grad_q, None, None


class XPUInterfaceManager(MegatronInterfaceManager):
    """XPUInterfaceManager"""

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
        self._custom_interfaces['custom_swiglu'] = CustomSwiGLUWrapper
        self._custom_interfaces['custom_rope'] = CustomRoPEWrapper(RotaryPositionalEmbeddingFunction)

    def set_inner_func(self, interface_name, inner_func):
        """set_inner_func"""
        global _INTERFACE_INNER_FUNCS
        _INTERFACE_INNER_FUNCS[interface_name] = inner_func

    def get_interface(self, interface_name):
        """get_interface"""
        return self._custom_interfaces[interface_name]
