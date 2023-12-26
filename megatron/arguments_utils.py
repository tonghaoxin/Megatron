#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments utils."""

import argparse
from typing import Dict
from functools import partial


class GroupArgsExtractor:
    """group args extractor"""
    def __init__(self):
        """__init__"""
        self._group_map_dict: Dict[str, argparse._ArgumentGroup] = {}

    @property
    def group_map(self):
        """get group map"""
        return self._group_map_dict

    @property
    def group_map_items(self):
        """get group map items"""
        return self._group_map_dict.items()

def get_extractor(parser: argparse.ArgumentParser):
    """get extractor"""
    if not hasattr(parser, "_extractor"):
        setattr(parser, "_extractor", GroupArgsExtractor())
    return getattr(parser, "_extractor")


def extract_group_args(group_name: str):
    """extract group args"""
    def _inner_extract_group_args(add_args_fn):
        """inner extract group args"""
        def _add_argument_group(extractor: GroupArgsExtractor,
                                add_group_fn: argparse.ArgumentParser.add_argument_group,
                                group_name: str, *args, **kwargs):
            """add argument group"""
            group_map = extractor.group_map
            if group_name in group_map:
                raise ValueError(
                    f"Duplicate group name received: {group_name}")
            new_group = add_group_fn(*args, **kwargs)
            group_map[group_name] = new_group
            return new_group

        def wrapper(parser: argparse.ArgumentParser):
            """wrapper for function"""
            add_group_fn = parser.add_argument_group
            extractor = get_extractor(parser)
            parser.add_argument_group = partial(
                _add_argument_group, extractor, add_group_fn, group_name)
            parser = add_args_fn(parser)
            parser.add_argument_group = add_group_fn
            return parser
        return wrapper
    return _inner_extract_group_args


def parse_group_args_to_dict(parser: argparse.ArgumentParser, args):
    """parse group args to dict"""
    extractor = get_extractor(parser)
    for name, group in extractor.group_map_items:
        if name in args:
            raise ValueError(f"Duplicate key found in args: {name}")
        group_args_dict = {action.dest: getattr(
            args, action.dest, None) for action in group._group_actions}
        setattr(args, name, group_args_dict)
    return args
