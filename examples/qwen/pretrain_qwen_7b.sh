#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
#DATA_PATH=/home/datasets/qwen/wudao_qwenbpe_content_document
DATA_PATH=/home/datasets/qwen_test/qwen_00_text_document_content_sentence
#CHECKPOINT_PATH=/ssd2/models/Qwen/Qwen-7B
CHECKPOINT_PATH=/ssd2/models/Qwen/Qwen-7B
NSYS_PATH=${ROOT_DIR}/../nsys/
TOKENIZER_PATH=/ssd2/models/Qwen/Qwen-7B/qwen.tiktoken \
TENSORBOARD_PATH=/workspace/tensorboard/new

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export CUDA_DEVICE_MAX_CONNECTIONS=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export CUDA_VISIBLE_DEVICES='4,5'

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

QWEN_7B_ARGS="--tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 1 \
               --num-layers 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 1 \
               --global-batch-size 1 \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --lr 1.0e-5 \
               --min-lr 1.0e-6 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --weight-decay 0.1 \
               --adam-eps 1e-08 \
               --train-iters 6400000 \
               --lr-decay-iters 6400000 \
               --lr-decay-style linear \
               --lr-warmup-iters 6400 \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-type QwenTokenizer \
               --tokenizer-model $TOKENIZER_PATH \
               --initial-loss-scale 16 \
               --swiglu \
               --disable-bias-linear \
               --add-qkv-bias True \
               --use-rotary-position-embeddings \
               --no-position-embedding \
               --untie-embeddings-and-output-weights \
               --use-flash-attn \
               --init-method-std 0.006 \
               --no-load-optim \
               --no-load-rng \
               --num-workers 8 \
               --seed 1234 \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --use-distributed-optimizer \
               --accumulate-allreduce-grads-in-fp32 \
               --bf16"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 98,2,0 \
            --tensorboard-dir $TENSORBOARD_PATH \
            --load $CHECKPOINT_PATH \
            --tensorboard-queue-size 1 \
            --log-timers-to-tensorboard \
            --log-batch-size-to-tensorboard \
            --log-validation-ppl-to-tensorboard \
            --distributed-backend nccl"
            # --save $CHECKPOINT_PATH \

#--normalization RMSNorm --position-embedding-type rope --norm-epsilon 1e-6

PYTHONPATH="$ROOT_DIR/../..":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../../pretrain_qwen.py \
       $QWEN_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
       # $NSYS_ARGS \
