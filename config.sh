#!/bin/bash

#export BKCL_RDMA_NICS=ens11np0,ens12np0,ens13np0,ens14np0,ens15np0,ens16np0,ens17np0,ens18np0  # 参考之前配置即可
export BKCL_RDMA_NICS=eth10,eth11,eth12,eth2,eth3,eth7,eth8,eth9
#export NCCL_SOCKET_IFNAME=ens11np0  # 参考之前配置即可
export NCCL_SOCKET_IFNAME=bond0.1061

export DATA_PATH=/workspace/llama2_70b_data/alpaca/alpaca_text_document 
export TOKENIZER_PATH=/workspace/llama2_70b_data/tokenizer.model
export CHECKPOINT_PATH=/workspace/checkpoints_4_4_2
export SO_PATH=/workspace/xre-Linux-x86_64-5.0.15.1/so:/workspace/xccl_rdma-ubuntu_x86_64/so/:$LD_LIBRARY_PATH

# 4机32卡配置
export NNODES=4
export NODE_RANK=0               ### 注意：各个机器分别设置为0,1,2,3
export MASTER_ADDR="192.168.0.15"  ### 注意：选择4机中1台作为master，4机的MASTER_ADDR全部设置为master结点的IP
export MASTER_PORT=9916          ### 设置一个可用端口
export EXIT_ITER=10
