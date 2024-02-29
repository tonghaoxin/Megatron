# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""NPU Custom Operators"""

import os
import pathlib
import subprocess

from torch.utils import cpp_extension
from .ascend_turbo.initialize import initialize_cfg_from_args


def load(args):
    """load"""
    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'
    _create_build_dir(buildpath)

    # Helper function to build pure C functions.
    def _c_extention_load_helper(name, sources):
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3',],
            verbose=(args.rank == 0)
        )

    # ============================
    # Attention update.
    # ============================
    sources = [srcpath / 'parallel_attention.cpp']
    parallel_attention = _c_extention_load_helper("parallel_attention", sources)


def _create_build_dir(buildpath):
    """_create_build_dir"""
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")
