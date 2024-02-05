# !/usr/bin/env python3

# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""MegatronCustomOperatorManager."""

import abc
from abc import ABC


class MegatronCustomOperatorManager(ABC):
    """MegatronCustomOperatorManager"""

    def __init__(self):
        """__init__"""
        self._custom_ops = None
        self._custom_interface_manager = None

    @abc.abstractmethod
    def _monkey_patch(self, args):
        """_monkey_patch"""
        ...

    @abc.abstractmethod
    def register_ops(self, args):
        """register_ops"""
        ...

    @abc.abstractmethod
    def get_operator(self, op_name):
        """get_operator"""
        ...
