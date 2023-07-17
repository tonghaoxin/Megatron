ARG BASE_IMAGE="iregistry.baidu-int.com/hac_test/nvcr.io/nvidia/pytorch:22.12-py3_cuda11.8"
FROM ${BASE_IMAGE}

# 流水线会根据编译输出的算子二进制打到上到bos上，版本参数是流水线 AGILE_PIPELINE_BUILD_ID 变量，每执行一次流水线都会变化
ARG BOS_BINARY_ADDR={{BOS_BINARY_ADDR}}
# 默认: false, 设置true 则替换编译算子二进制
ARG BINARY_REPLACE="false"

RUN rm -rf /workspace/
COPY Megatron /workspace/Megatron-LM
RUN bash /workspace/Megatron-LM/docker/install.sh ${BOS_BINARY_ADDR} ${BINARY_REPLACE}