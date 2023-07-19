#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../data/openwebtext-10k_text_document
CHECKPOINT_SAVE_PATH=${ROOT_DIR}/../checkpoints_to_save/
CHECKPOINT_LOAD_PATH=${ROOT_DIR}/../checkpoints_to_load/
TOKENIZER_PATH=${ROOT_DIR}/../tokenizer

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_MAX_CONNECTIONS=1


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

BLOOM_7B_ARGS="--tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 1 \
               --num-layers 30 \
               --hidden-size 4096 \
               --num-attention-heads 32 \
               --micro-batch-size 4 \
               --global-batch-size 64 \
               --seq-length 512 \
               --max-position-embeddings 512 \
               --lr 2e-5 \
               --min-lr 1e-6 \
               --clip-grad 1.0 \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --weight-decay 1e-4 \
               --train-iters 500000 \
               --lr-decay-iters 320000 \
               --lr-decay-style cosine \
               --lr-warmup-fraction 0 \
               --tokenizer-type BloomHFTokenizer \
               --hf-tokenizer-path $TOKENIZER_PATH \
               --embedding-dropout 0 \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --use-alibi-position-embeddings \
               --activation-func openai_gelu \
               --pad-vocab-size-to 250880 \
               --layernorm-epsilon 1e-05 \
               --add-embed-layernorm \
               --use-distributed-optimizer \
               --no-query-key-layer-scaling \
               --no-position-embedding \
               --no-load-optim \
               --no-load-rng \
               --initial-loss-scale 16 \
               --bf16"
               # --fp16"
               # --sequence-parallel \
               # --attention-softmax-in-fp32 \

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--save $CHECKPOINT_SAVE_PATH \
            --load $CHECKPOINT_LOAD_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"

PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_bloom.py \
       $BLOOM_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
