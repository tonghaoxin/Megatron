#!/bin/bash
set -eo pipefail

# 将容器中编译好的算子压缩包

cd /workspace/

cuda_ops_build_folder="/workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/build"

if [ -d "$cuda_ops_build_folder" ]; then
    echo "cuda_ops_build_folder exists"
    cp /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/build/*.so ./
else
    echo "cuda_ops_build_folder not exists"
fi

cp /workspace/Megatron-LM/megatron/data/helpers.cpython-*.so ./
tar -czf megatron_compile.tar.gz *.so
