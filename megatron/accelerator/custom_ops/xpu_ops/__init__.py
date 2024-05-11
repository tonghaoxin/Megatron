# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""XPU Custom Operators"""

import os
import pathlib
import subprocess

from torch.utils import cpp_extension


def load_module(mod):
    """load_module"""
    if mod == "adam":
        from apex.optimizers import FusedAdam
        return FusedAdam
    elif mod == "sgd":
        from apex.optimizers import FusedSGD
        return FusedSGD
    elif mod == "fused_weight_gradient_mlp_cuda":
        import fused_weight_gradient_mlp_cuda
        return fused_weight_gradient_mlp_cuda
    elif mod == "flashattn":
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        return flash_attn_varlen_func
    elif mod == "multi_tensor_apply":
        from apex.multi_tensor_apply import multi_tensor_applier
        from amp_C import multi_tensor_l2norm
        from amp_C import multi_tensor_scale
        return multi_tensor_applier, multi_tensor_l2norm, multi_tensor_scale


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
