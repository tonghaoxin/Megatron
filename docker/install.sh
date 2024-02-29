#!/bin/bash

##########################################
#
# 1. 将项目copy 到容器，安装依赖
# 2. 增加参数判断是否需要替换二进制
#
##########################################
set -exuo pipefail

CURRENT_DIR=$(cd `dirname $0`; pwd)
BCCL_VERSION="1.1.4.1"
IREPO_TOKEN="dd785b53-3185-4772-8ae4-984f93a4c32c"
BCCL_DOWNLOAD_ADDR="wget -O output.tar.gz --no-check-certificate --header "IREPO-TOKEN:${IREPO_TOKEN}" "https://irepo.baidu-int.com/rest/prod/v3/baidu/AIPod/BCCL/releases/${BCCL_VERSION}/files""

############################################################# 基础环境开始 ###########################################################

function install_pip_config() {
    cp pip.conf /etc/pip.conf
    cp pip.conf /usr/pip.conf
    mkdir -p /root/.pip && cp pip.conf /root/.pip/pip.conf
    mkdir -p /etc/xdg/pip && cp pip.conf /etc/xdg/pip/pip.conf
}

function install_base_env() {
    local COMPILE_ENV
    read COMPILE_ENV<<< $*

    cd ${CURRENT_DIR}
    rm -rf /opt/conda/pip.conf /root/.config/pip/pip.conf /root/.pip/pip.conf /etc/pip.conf /etc/xdg/pip/pip.conf /usr/pip.conf

    requirements_file="requirements.txt"
    if  [[ "$COMPILE_ENV" == "npu" ]];then
        requirements_file="requirements_npu.txt"
    fi

    pip install -r /workspace/Megatron-LM/${requirements_file} -i http://mirrors.baidubce.com/pypi/simple --trusted-host mirrors.baidubce.com

    # pip config
    install_pip_config

    rm -rf /etc/apt/sources.list && cp /workspace/Megatron-LM/docker/sources.list /etc/apt/sources.list

    # 设置时区
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install tzdata -y && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    apt-get install bc -y
}

function clear_unused_file() {
    # 删除 /tmp/* ~/.bash_history
    rm -rf /tmp/* ~/.bash_history

    # 删除git相关
    rm -rf /workspace/Megatron-LM/.git

    # 删除不对外暴露的文件
    rm -rf /workspace/Megatron-LM/build.sh
    rm -rf /workspace/Megatron-LM/ci.yml
    rm -rf /workspace/Megatron-LM/docker
    rm -rf /workspace/Megatron-LM/ci
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
    rm -rf /workspace/Megatron-LM/megatron/accelerator/custom_ops/xpu_ops/*
    rm -rf /workspace/Megatron-LM/megatron/accelerator/custom_ops/npu_ops/*
    mv /workspace/Megatron-LM/docker/__init__.py /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/__init__.py
    cp ${CURRENT_DIR}/${compile_folder_name}/parallel_attention.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/scaled_masked_softmax_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/scaled_softmax_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/scaled_upper_triang_masked_softmax_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/rotary_positional_embedding_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/
    cp ${CURRENT_DIR}/${compile_folder_name}/matmul_reduce_parallel_cuda.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/

    # 替换 data 下的 helpers.cpython-38-x86_64-linux-gnu.so 二进制
    cp ${CURRENT_DIR}/${compile_folder_name}/helpers.cpython-*.so /workspace/Megatron-LM/megatron/data/

    # 删除hybrid_adam目录下的builder和csrc源码，安装cpu_adam.so
    rm -rf /workspace/Megatron-LM/megatron/hybrid_adam/builder
    rm -rf /workspace/Megatron-LM/megatron/hybrid_adam/csrc
    cp ${CURRENT_DIR}/${compile_folder_name}/cpu_adam.so /workspace/Megatron-LM/megatron/hybrid_adam/

    rm -rf ${CURRENT_DIR}/${compile_folder_name}.tar.gz ${CURRENT_DIR}/${compile_folder_name}
}

function download_megatron_binary_compile_for_npu() {
    # 背景是：由于每次启动 Megatron 训练任务都会编译一次，并且编译时间还比较长/编译机器时常没有，现增加编译好的二进制进行替换逻辑;

    # "a800"，使用a800编译的二进制
    local COMPILE_ENV
    read COMPILE_ENV<<< $*
    cd ${CURRENT_DIR}
    compile_folder_name="megatron_compile"
    wget https://cce-ai-datasets.bj.bcebos.com/hac_test/megatron/binary_compile/${COMPILE_ENV}/${compile_folder_name}.tar.gz
    mkdir -p ${compile_folder_name}
    tar -zxvf ${compile_folder_name}.tar.gz -C ${compile_folder_name}

    # 替换 __init__NPU.py 和 cuda_ops 下的二进制
    rm -rf /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/*
    rm -rf /workspace/Megatron-LM/megatron/accelerator/custom_ops/xpu_ops/*
    rm -rf /workspace/Megatron-LM/megatron/accelerator/custom_ops/npu_ops/*
    mv /workspace/Megatron-LM/docker/__init__NPU.py /workspace/Megatron-LM/megatron/accelerator/custom_ops/cuda_ops/__init__.py
    cp ${CURRENT_DIR}/${compile_folder_name}/parallel_attention.so /workspace/Megatron-LM/megatron/accelerator/custom_ops/npu_ops/

    # 替换 data 下的 helpers.cpython-38-x86_64-linux-gnu.so 二进制
    cp ${CURRENT_DIR}/${compile_folder_name}/helpers.cpython-*.so /workspace/Megatron-LM/megatron/data/

    # 删除hybrid_adam目录下的builder和csrc源码，安装cpu_adam.so
    rm -rf /workspace/Megatron-LM/megatron/hybrid_adam/builder
    rm -rf /workspace/Megatron-LM/megatron/hybrid_adam/csrc
    cp ${CURRENT_DIR}/${compile_folder_name}/cpu_adam.so /workspace/Megatron-LM/megatron/hybrid_adam/

    rm -rf ${CURRENT_DIR}/${compile_folder_name}.tar.gz ${CURRENT_DIR}/${compile_folder_name}

}

function encrypt_code() {
    cd  /workspace/Megatron-LM
    cp tools/cython_setup.py ./
    python cython_setup.py build
    rm -rf cython_setup.py tools/cython_setup.py
}

function install_bccl_env() {
    local BCCL_DOWNLOAD_ADDR=$1
    cd /workspace
    eval "$BCCL_DOWNLOAD_ADDR"
    tar -zxvf output.tar.gz
    cd ./output/nccl_build3/aiak-megatron-a800
    dpkg -r libnccl2 libnccl-dev  # 这个用来卸载 aiak 老的包
    dpkg -i bccl-*.amd64.deb

    rm -rf /workspace/output*
}

# 参数1：编译GPU环境, h800/a800/npu
# 参数2：是否替换二进制
# 参数3：是否对代码加密
# 参数4：是否安装BCCL环境
# 参数5：BCCL下载地址
# 参数6：是否安装NPU环境
COMPILE_ENV=$1
BINARY_REPLACE=$2
ENCRYPT_FLAG=$3
INSTALL_BCCL_FLAG=$4
INSTALL_BCCL_ADDR=$5
NPU_FLAG=$6

if [[ "$INSTALL_BCCL_ADDR" != "" ]];then
    BCCL_DOWNLOAD_ADDR=$INSTALL_BCCL_ADDR
fi

if  [[ "$ENCRYPT_FLAG" == "true" ]];then
    encrypt_code
fi

if  [[ "$BINARY_REPLACE" == "true" ]];then
    download_megatron_binary_compile $COMPILE_ENV
elif  [[ "$BINARY_REPLACE" == "all" ]];then
    install_base_env $COMPILE_ENV
    download_megatron_binary_compile $COMPILE_ENV
elif  [[ "$BINARY_REPLACE" == "false" ]];then
    install_base_env $COMPILE_ENV
fi

if  [[ "$INSTALL_BCCL_FLAG" == "true" ]];then
    install_bccl_env "$BCCL_DOWNLOAD_ADDR"
fi

if  [[ "$NPU_FLAG" == "true" ]];then
    echo "NPU install start ..."
    download_megatron_binary_compile_for_npu a800
    echo "NPU install end"
fi

# 清理无用文件
clear_unused_file

############################################################# 基础环境结束 ###########################################################
