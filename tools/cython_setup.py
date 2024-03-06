# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Megatron Cython Build"""

import os
import sys
import shutil
import numpy
import tempfile
import argparse

from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython.Compiler import Options
Options.error_on_unknown_names = False

import platform

from distutils.core import setup
from Cython.Build import cythonize

group_num = 2
file_groups = []


v3_list = ["megatron/checkpoint_manager.py",
           "megatron/optimizer/optimizer.py",
           "megatron/optimizer/distrib_optimizer.py",
           "megatron/optimizer/zero_optimizer.py",
           "megatron/optimizer/clip_grads.py",
           "megatron/optimizer/chunk/chunk.py",
           "megatron/optimizer/chunk/chunk_mgr.py",
           "megatron/optimizer/memory_tracer/memory_monitor.py",
           "megatron/optimizer/memory_tracer/memory_stats.py",
           "megatron/hybrid_adam/hybrid_adam.py",
           "megatron/hybrid_adam/multi_thread_hybrid_adam.py",
           "megatron/accelerator/abstract_accelerator.py",
           "megatron/accelerator/cuda_accelerator.py",
           "megatron/accelerator/xpu_accelerator.py",
           "megatron/accelerator/npu_accelerator.py",
           "megatron/accelerator/custom_ops/abstract_custom_operator_manager.py",
           "megatron/accelerator/custom_ops/cuda_custom_operator_manager.py",
           "megatron/accelerator/custom_ops/xpu_custom_operator_manager.py",
           "megatron/accelerator/custom_ops/npu_custom_operator_manager.py",
           "megatron/accelerator/custom_ops/npu_ops/ascend_turbo/ascend_turbo_cfg.py",
           "megatron/accelerator/custom_ops/npu_ops/ascend_turbo/initialize.py",
           "megatron/accelerator/custom_ops/npu_ops/ascend_turbo/mc2_linears_seq_parallel.py",
           "megatron/accelerator/interfaces/abstract_interface_manager.py",
           "megatron/accelerator/interfaces/cuda_interface_manager.py",
           "megatron/accelerator/interfaces/xpu_interface_manager.py",
           "megatron/accelerator/interfaces/npu_interface_manager.py",
           "megatron/core/parallel_state.py",
           "megatron/core/pipeline_parallel/p2p_communication.py",
           "megatron/core/pipeline_parallel/utils.py",
           "megatron/core/tensor_parallel/mappings.py",
           "megatron/core/tensor_parallel/cross_entropy.py",
           "megatron/core/tensor_parallel/data.py"]


v2_list = []


def build_file_groups():
    """build file groups for cython build"""
    group_dict_v3 = dict()
    group_dict_v2 = dict()

    group_dict_v3.update({"file_list": v3_list})
    group_dict_v3.update({"version": "3"})
    file_groups.append(group_dict_v3)


    group_dict_v2.update({"file_list": v2_list})
    group_dict_v2.update({"version": "2"})
    file_groups.append(group_dict_v2)

def exec_cython_build():
    """exec cython build"""
    for file_group in file_groups:
        version = file_group["version"]
        for file in file_group["file_list"]:
            if not os.path.exists(file):
               continue
            if not file.endswith(".py"):
               print ("%s not python source file"%file)
               continue

            pure_file_name = file[:-3]
            dir_path = os.path.dirname(file)

            setup(name=pure_file_name, ext_modules=cythonize(file, compiler_directives={'language_level' : version}))

            os.system("rm -rf %s.py"%pure_file_name)
            os.system("rm -rf %s.c"%pure_file_name)
            os.system("cp build/lib.linux-x86_64-*/%s* %s"%(pure_file_name, dir_path))

def clear_build_dir():
    """rm build dir"""
    if os.path.exists("./build"):
       os.system("rm -rf ./build")

def show_help():
    """show help infomations."""
    print("python cython_setup.py -h or --help: show help infomations.")
    print("python cython_setup.py build: exec cython build.")

def main():
    """main"""
    if len(sys.argv) < 2:
        print (f"Number of command line parameters is not right,"
               f"should add 'build', like 'python cython_setup.py build'")
        return

    if len(sys.argv) > 2:
        print ("Number of command line parameters is not right, more than the normal numbers")
        return

    if sys.argv[1] != "build" and sys.argv[1] != "-h" and sys.argv[1] != "--help":
        print ("The last parameter is not right, should be build,or -h,or --help.")
        return

    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        show_help()
        return

    build_file_groups()
    exec_cython_build()
    clear_build_dir()

if __name__ == '__main__':
    main()
