ARG BASE_IMAGE="registry.baidubce.com/cce-ai-native/aiak-megatron:v1.2.4.3_ubuntu20.04-cu11.8-torch1.14.0-py38"
# ARG BASE_IMAGE="registry.baidubce.com/cce-ai-native/aiak-megatron:v1.2.4.3_ubuntu22.04-cu12.2-torch2.1.0-py310"
FROM ${BASE_IMAGE}

RUN rm -rf /workspace/Megatron-LM/ci/gpu/scripts
RUN mkdir -p /workspace/Megatron-LM/ci/gpu
COPY . /workspace/Megatron-LM/ci/gpu/scripts