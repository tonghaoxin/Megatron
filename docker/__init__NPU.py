#!/usr/bin/env python3
"""init"""
from .ascend_turbo.initialize import initialize_cfg_from_args
def load(args):
    """load"""
    import parallel_attention
