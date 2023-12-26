#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../data/baichuan_00_text_document
CHECKPOINT_PATH=${ROOT_DIR}/../checkpoints/
TOKENIZER_PATH=${ROOT_DIR}/../tokenizer.model
NSYS_PATH=${ROOT_DIR}/../nsys/

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

NSYS_ARGS="nsys profile \
           --trace=cuda,nvtx,osrt \
           --gpu-metrics-device=0,1,2,3,4,5,6,7 \
           --gpu-metrics-frequency=20000 \
           --gpu-metrics-set=ga100 \
           --delay=30 \
           --duration=30 \
           --force-overwrite true \
           -o $NSYS_PATH/node_$NODE_RANK"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

BAICHUAN_7B_ARGS="--tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 1 \
               --num-layers 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 1 \
               --global-batch-size 1024 \
               --seq-length 4096 \
               --max-position-embeddings 4096 \
               --lr 1.0e-8 \
               --min-lr 1.0e-8 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --weight-decay 0.1 \
               --adam-eps 1e-08 \
               --train-iters 500000 \
               --lr-decay-iters 320000 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .01 \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-type BaichuanTokenizer \
               --tokenizer-model $TOKENIZER_PATH \
               --activation-func swiglu \
               --use-rotary-position-embeddings \
               --apply-rotary-positional-embedding-kernel \
               --rmsnorm-epsilon 1e-6 \
               --fused-rmsnorm \
               --no-position-embedding \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --untie-embeddings-and-output-weights \
               --initial-loss-scale 16 \
               --use-flash-attn \
               --use-distributed-optimizer \
               --num-workers 8 \
               --fp16"

# Alibi is not supported by FlashAttention for now.
BAICHUAN_13B_ARGS="--tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 1 \
                --num-layers 40 \
                --hidden-size 5120 \
                --ffn-hidden-size 13696 \
                --num-attention-heads 40 \
                --micro-batch-size 1 \
                --global-batch-size 1024 \
                --seq-length 4096 \
                --max-position-embeddings 4096 \
                --lr 1.0e-8 \
                --min-lr 1.0e-8 \
                --clip-grad 1.0 \
                --optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.95 \
                --weight-decay 0.1 \
                --adam-eps 1e-08 \
                --train-iters 500000 \
                --lr-decay-iters 320000 \
                --lr-decay-style cosine \
                --lr-warmup-fraction .01 \
                --recompute-method uniform \
                --no-async-tensor-model-parallel-allreduce \
                --fp16
                --tokenizer-type BaichuanTokenizer \
                --tokenizer-model $TOKENIZER_PATH \
                --activation-func swiglu \
                --rmsnorm-epsilon 1e-6 \
                --fused-rmsnorm \
                --no-position-embedding \
                --disable-bias-linear \
                --attention-dropout 0 \
                --hidden-dropout 0 \
                --embedding-dropout 0 \
                --untie-embeddings-and-output-weights \
                --use-alibi-position-embeddings \
                --num-workers 8"
                # --sequence-parallel

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"
            # --load $CHECKPOINT_PATH \
            # --save $CHECKPOINT_PATH \

PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_baichuan.py \
       $BAICHUAN_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
       # $NSYS_ARGS \
