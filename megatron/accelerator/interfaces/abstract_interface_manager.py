# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""MegatronInterfaceManager."""

import abc
from abc import ABC


# Dict to store { interface_name: inner_func }
_INTERFACE_INNER_FUNCS = {}


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
