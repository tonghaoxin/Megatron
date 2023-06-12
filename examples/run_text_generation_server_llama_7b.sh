#!/bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
CHECKPOINT_PATH=${ROOT_DIR}/../checkpoints/megatron_llama_7b_checkpoint_tp2_pp2

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

export CUDA_DEVICE_MAX_CONNECTIONS=1

PYTHONPATH="$ROOT_DIR/..":"$ROOT_DIR/../megatron/fused_kernels":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun $DISTRIBUTED_ARGS $ROOT_DIR/../tools/run_llama_text_generation_server.py   \
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --disable-bias-linear \
       --num-attention-heads 32 \
       --max-position-embeddings 1024  \
       --no-position-embedding \
       --tokenizer-type LLaMASentencePieceTokenizer \
       --tokenizer-model $ROOT_DIR/../download/tokenizer.model \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --out-seq-length 1024  \
       --tensor-model-parallel-size 2  \
       --pipeline-model-parallel-size 2  \
       --load ${CHECKPOINT_PATH}  \
       --swiglu \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --embedding-dropout 0 \
       --fused-rmsnorm \
       --rmsnorm-epsilon 1e-6 \
       --seed 42