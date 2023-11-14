#!/bin/bash

##########################################
#
# 1. 将项目copy 到容器，安装依赖
# 2. 增加参数判断是否需要替换二进制
#
##########################################
set -exuo pipefail

CURRENT_DIR=$(cd `dirname $0`; pwd)

############################################################# 基础环境开始 ###########################################################

function install_pip_config() {
    cp pip.conf /etc/pip.conf
    cp pip.conf /usr/pip.conf
    mkdir -p /root/.pip && cp pip.conf /root/.pip/pip.conf
    mkdir -p /etc/xdg/pip && cp pip.conf /etc/xdg/pip/pip.conf
}

function install_base_env() {
    cd ${CURRENT_DIR}
    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf
    pip install -r /workspace/Megatron-LM/requirements.txt -i http://mirrors.baidubce.com/pypi/simple --trusted-host mirrors.baidubce.com

    # pip config
    install_pip_config

    rm -rf /etc/apt/sources.list && cp /workspace/Megatron-LM/docker/sources.list /etc/apt/sources.list

    # 设置时区
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install bc

    # 删除 /tmp/* ~/.bash_history
    rm -rf /tmp/* ~/.bash_history

    # 删除git相关
    rm -rf /workspace/Megatron-LM/.git
}

function download_megatron_binary_compile() {
    # 背景是：由于每次启动 Megatron 训练任务都会编译一次，并且编译时间还比较长/编译机器时常没有，现增加编译好的二进制进行替换逻辑;

    # "h800"、"a800"
    local COMPILE_ENV
    read COMPILE_ENV<<< $*
    cd ${CURRENT_DIR}
    compile_folder_name="megatron_compile"
    wget https://cce-ai-datasets.bj.bcebos.com/hac_test/megatron/binary_compile/${COMPILE_ENV}/${compile_folder_name}.tar.gz
    mkdir -p ${compile_folder_name}
    tar -zxvf ${compile_folder_name}.tar.gz -C ${compile_folder_name}

    # 替换 __init__.py 和 cuda_ops 下的二进制
    rm -rf /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/*
    mv /workspace/Megatron-LM/docker/__init__.py /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/__init__.py
    cp ${CURRENT_DIR}/${compile_folder_name}/parallel_attention.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/scaled_masked_softmax_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/scaled_softmax_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/scaled_upper_triang_masked_softmax_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/rotary_positional_embedding_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/matmul_reduce_parallel_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/

    # 替换 data 下的 helpers.cpython-38-x86_64-linux-gnu.so 二进制
    cp ${CURRENT_DIR}/${compile_folder_name}/helpers.cpython-*.so /workspace/Megatron-LM/megatron/data/

    rm -rf ${CURRENT_DIR}/${compile_folder_name}.tar.gz ${CURRENT_DIR}/${compile_folder_name}
}

# 参数1：编译GPU环境, h800/a800
# 参数2：是否替换二进制
COMPILE_ENV=$1
BINARY_REPLACE=$2

if  [[ "$BINARY_REPLACE" == "true" ]];then
    download_megatron_binary_compile $COMPILE_ENV
elif  [[ "$BINARY_REPLACE" == "all" ]];then
    install_base_env
    download_megatron_binary_compile $COMPILE_ENV
else
    install_base_env
fi



############################################################# 基础环境结束 ###########################################################
