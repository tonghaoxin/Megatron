#!/bin/bash

###
#
# 1. 将项目copy 到容器，安装依赖
# 2. 增加参数判断是否需要替换二进制
#
###
set -exuo pipefail

CURRENT_DIR=$(cd `dirname $0`; pwd)

############################################################# 基础环境开始 ###########################################################

function install_base_env() {
    cd ${CURRENT_DIR}
    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf
    pip install -r /workspace/Megatron-LM/requirements.txt -i http://mirrors.baidubce.com/pypi/simple --trusted-host mirrors.baidubce.com
    cp pip.conf /etc/pip.conf
    rm -rf /etc/apt/sources.list && cp /workspace/Megatron-LM/docker/sources.list /etc/apt/sources.list

    # 设置时区
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install bc
}


function download_megatron_binary_compile() {
    # 背景是：由于每次启动 Megatron 训练任务都会编译一次，并且编译时间还比较长，所以将编译好的二进制进行替换，缩短训练时长;

    local BOS_BINARY_ADDR
    read BOS_BINARY_ADDR<<< $*
    cd ${CURRENT_DIR}
    wget ${BOS_BINARY_ADDR}
    mkdir -p ${CURRENT_DIR}/megatron_compile
    tar -zxvf megatron_compile.tar.gz -C ${CURRENT_DIR}/megatron_compile
    
    # 替换 __init__.py 和 fused_kernels下的二进制
    mv /workspace/Megatron-LM/megatron/fused_kernels/__init__.py /workspace/Megatron-LM/megatron/fused_kernels/__init__.py.bak‘
    cp ${CURRENT_DIR}/megatron_compile/__init__.py  /workspace/Megatron-LM/megatron/fused_kernels/__init__.py
    cp ${CURRENT_DIR}/megatron_compile/parallel_attention.so /workspace/Megatron-LM/megatron/fused_kernels/
    cp ${CURRENT_DIR}/megatron_compile/scaled_masked_softmax_cuda.so /workspace/Megatron-LM/megatron/fused_kernels/
    cp ${CURRENT_DIR}/megatron_compile/scaled_softmax_cuda.so /workspace/Megatron-LM/megatron/fused_kernels/
    cp ${CURRENT_DIR}/megatron_compile/scaled_upper_triang_masked_softmax_cuda.so /workspace/Megatron-LM/megatron/fused_kernels/

    # 替换 data 下的 helpers.cpython-38-x86_64-linux-gnu.so 二进制
    cp ${CURRENT_DIR}/megatron_compile/helpers.cpython-38-x86_64-linux-gnu.so /workspace/Megatron-LM/megatron/data/helpers.cpython-38-x86_64-linux-gnu.so
}

if  [[ "$2" == "true" ]];then
    download_megatron_binary_compile $1
else
    install_base_env
fi



############################################################# 基础环境结束 ###########################################################
