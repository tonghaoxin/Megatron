"""
    CPUAdamBuilder used for compiling to the dynamic library of cpu_adam.so
"""
import os
from .builder import TorchCPUOpBuilder


class CPUAdamBuilder(TorchCPUOpBuilder):
    """
    CPUAdamBuilder used for compiling to the dynamic library of cpu_adam.so
    """
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"

    def __init__(self):
        """
        Init the  TorchCPUOpBuilder.
        """
        super().__init__(name=self.NAME)

    def absolute_name(self):
        """
        Returns the ops of cpu_adam.
        """
        return ('megatron.ops.adam.%s_op')%self.NAME

    def sources(self):
        """
        Returns the sources of cpu_adam.
        """
        if self.build_for_cpu:
            return ['csrc/adam/cpu_adam.cpp']

        return ['csrc/adam/adam_task.cpp', 'csrc/adam/adam_thread.cpp', \
                'csrc/adam/cpu_adam.cpp', 'csrc/common/custom_cuda_kernel.cu']

    def libraries_args(self):
        """
        Returns list of the libraries.
        """
        args = super().libraries_args()
        if self.build_for_cpu:
            return args

        if not self.is_rocm_pytorch():
            args += ['curand']

        return args

    def include_paths(self):
        """
        Returns list of include paths, relative to root of Megatron package
        """
        import torch
        if self.build_for_cpu:
            CUDA_INCLUDE = []
        elif not self.is_rocm_pytorch():
            CUDA_INCLUDE = [os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")]
        else:
            CUDA_INCLUDE = [
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include"),
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include", "rocrand"),
                os.path.join(torch.utils.cpp_extension.ROCM_HOME, "include", "hiprand"),
            ]
        return ['csrc/includes'] + CUDA_INCLUDE
