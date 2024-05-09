ARG BASE_IMAGE="iregistry.baidu-int.com/hac_test/nvcr.io/nvidia/pytorch:22.12-py3_cuda11.8"
# A800: iregistry.baidu-int.com/hac_test/nvcr.io/nvidia/pytorch:22.12-py3_cuda11.8
# H800: registry.baidubce.com/cce-ai-native/nvidia/pytorch:23.10-py3
# NPU:  registry.baidubce.com/cce-ai-native/aiak-megatron:multi-accelerators-npu_v1

FROM ${BASE_IMAGE}

# 支持根据卡类型获取算子包
ARG COMPILE_ENV={{COMPILE_ENV}}

# 默认: false, 设置true 则替换编译算子二进制
ARG BINARY_REPLACE="false"

# 默认: false, 设置true 是否需要针对代码进行加密
ARG ENCRYPT_FLAG="false"

# 默认: false, 设置true 是否安装bccl
ARG INSTALL_BCCL_FLAG="false"

# NPU 是否支持标志
ARG NPU_FLAG="false"

# XPU 是否支持标志
ARG XPU_FLAG="false"

# bccl 下载地址
ARG INSTALL_BCCL_ADDR=""

# 标志是否是BCCL 版本
ENV USE_BCCL=${INSTALL_BCCL_FLAG}

ENV LD_LIBRARY_PATH=/usr/local/python3.8.18/lib:$LD_LIBRARY_PATH
ENV PATH /usr/local/python3.8.18/bin:$PATH

RUN rm -rf /workspace/ && mkdir -p /workspace
COPY Megatron /workspace/Megatron-LM
RUN bash /workspace/Megatron-LM/docker/install.sh ${COMPILE_ENV} ${BINARY_REPLACE} ${ENCRYPT_FLAG} ${INSTALL_BCCL_FLAG} "${INSTALL_BCCL_ADDR}" "${NPU_FLAG}" "${XPU_FLAG}"