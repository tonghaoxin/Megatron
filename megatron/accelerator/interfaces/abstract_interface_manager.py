# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""MegatronInterfaceManager."""

import torch

import abc
from abc import ABC


# Dict to store { interface_name: inner_func }
_INTERFACE_INNER_FUNCS = {}


def _rotate_half_v1(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_v2(x):
    """
    another version of _rotate_half
    """
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    # dim=-1 triggers a bug in earlier torch versions
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


class MegatronInterfaceManager(ABC):
    """MegatronInterfaceManager"""

    def __init__(self):
        """__init__"""
        self._custom_interfaces = None

    @abc.abstractmethod
    def set_inner_func(self, interface_name, inner_func):
        """set_inner_func"""
        ...

    @abc.abstractmethod
    def get_interface(self, interface_name):
        """get_interface"""
        ...
