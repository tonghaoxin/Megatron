"""Activation function module"""

import torch
from torch import nn
from torch.nn import functional as F
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.accelerator import get_accelerator


class _ActivationBaseModule(nn.Module):
    """Activation base module"""
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        r"""Applies the activation function to the incoming data.

        Args:
            x (torch.Tensor): The input tensor.
        """
        return self.activation_fn(x)


class _GLUBaseModule(_ActivationBaseModule):
    """GLU base module"""
    def __init__(self, activation_fn):
        super().__init__(activation_fn)

    def forward(self, x):
        r"""Applies the GLU-based activation function to the incoming data.

        Args:
            x (torch.Tensor): The input tensor.
        """
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


class LiGLU(_GLUBaseModule):
    """LiGLU module"""
    def __init__(self):
        super().__init__(nn.Identity())


class GeGLU(_GLUBaseModule):
    """GeGLU module"""
    def __init__(self):
        super().__init__(F.gelu)


class ReGLU(_GLUBaseModule):
    """ReGLU module"""
    def __init__(self):
        super().__init__(F.relu)


class SwiGLU(_GLUBaseModule):
    """SwiGLU module"""
    def __init__(self):
        super().__init__(F.silu)

    def forward(self, x):
        r"""Applies the SwiGLU activation function to the incoming data.

        Args:
            x (torch.Tensor): The input tensor.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return self.activation_fn(x1) * x2


class OpenAIGeLU(_ActivationBaseModule):
    """OpenAIGeLU module

    Note:
        This module should not be used unless for backward compatibility reasons.
    """
    def __init__(self):
        super().__init__(openai_gelu)


class ErfGeLU(_ActivationBaseModule):
    """ErfGeLU module

    Note:
        This module should be used as workarounds for known problems with Torch ONNX exporter.
    """
    def __init__(self):
        super().__init__(erf_gelu)


class SquaredReLU(_ActivationBaseModule):
    """SquaredReLU module"""
    def __init__(self):
        super().__init__(F.relu)

    def forward(self, x):
        r"""Applies the SquaredReLU function to the incoming data.

        Args:
            x (torch.Tensor): The input tensor.
        """
        return torch.pow(self.activation_fn(x), 2)


relu = F.relu
gelu = F.gelu
liglu = LiGLU()
geglu = GeGLU()
reglu = ReGLU()
swiglu = SwiGLU()
openai_gelu = OpenAIGeLU()
erf_gelu = ErfGeLU()
squared_relu = SquaredReLU()


# interface for swiglu replacement
custom_swiglu = get_accelerator().get_operator('custom_swiglu')
if custom_swiglu is not None:
    swiglu = custom_swiglu


ACTIVATION_FUNCS = {
    "relu": relu,
    "gelu": gelu,
    "geglu": geglu,
    "liglu": liglu,
    "reglu": reglu,
    "swiglu": swiglu,
    "openai_gelu": openai_gelu,
    "erf_gelu": erf_gelu,
    "squared_relu": squared_relu,
}
