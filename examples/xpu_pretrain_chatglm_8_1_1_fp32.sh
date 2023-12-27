#! /bin/bash

NAME="ChatGLM-6B"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=/megatron_datasets/chatglm/data/wudao/
CHECKPOINT_PATH=/megatron_datasets/chatglm/checkpoints/tp8-pp1-dp1-fp32/
TOKENIZER_MODEL_PATH=/megatron_datasets/chatglm/tokenizer_model/ice_text.model
TENSORBOARD_PATH=./tensorborads/$EXP_NAME
LOG_PATH=./logs/$EXP_NAME

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9988}
NNODES=${1:-1}
NODE_RANK=${2:-0}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

NLAYERS=28
NHEADS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=16384

GLM_ARGS="--num-layers $NLAYERS \
          --hidden-size $HIDDEN_SIZE \
          --ffn-hidden-size $FFN_HIDDEN_SIZE \
          --num-attention-heads $NHEADS \
          --make-vocab-size-divisible-by 768 \
          --positional-encoding-2d \
          --rotary-embedding-2d \
          --model-type-custom chatglm-6b \
          --use-rotary-position-embeddings-with-position-ids \
          --seq2seq-data-collator \
          --deepnorm"

MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=128

TP_SIZE=8
PP_SIZE=1
SEQ_LEN=1024

OPTIMIZER_ARGS="--optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.995 \
                --adam-eps 1e-8 \
                --lr 1e-5 \
                --min-lr 1e-6 \
                --override-opt_param-scheduler \
                --lr-decay-style cosine \
                --clip-grad 1.0 \
                --weight-decay 1e-1"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 3 \
             --tensorboard-dir $TENSORBOARD_PATH \
             --tensorboard-queue-size 5 \
             --log-timers-to-tensorboard \
             --log-batch-size-to-tensorboard \
             --log-validation-ppl-to-tensorboard"

TRAINING_ARGS="--train-iters 5000 \
               --lr-decay-iters 3200 \
               --lr-warmup-fraction 0.01 \
               --tensor-model-parallel-size $TP_SIZE \
               --pipeline-model-parallel-size $PP_SIZE \
               --micro-batch-size $MICRO_BATCH_SIZE \
               --global-batch-size $GLOBAL_BATCH_SIZE \
               --seq-length $SEQ_LEN \
               --preprocessing-num-workers 1 \
               --load $CHECKPOINT_PATH \
               --exit-interval 2 \
               --data-path $DATA_PATH \
               --split 949,50,1 \
               --distributed-backend nccl \
               --init-method-std 0.0052 \
               --no-async-tensor-model-parallel-allreduce \
               --no-position-embedding \
               --no-gradient-accumulation-fusion \
               --no-masked-softmax-fusion \
               --tokenizer-model $TOKENIZER_MODEL_PATH \
               --initial-loss-scale 16 \
               --loss-scale 0 \
               --hidden-dropout 0 \
               --dataloader-type cyclic \
               --attention-dropout 0 \
               --embedding-dropout 0 \
               --no-contiguous-buffers-in-local-ddp"

OPTIONS_NCCL="NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_DISABLE=0"

#DISTRIBUTED_ARGS="--hostfile $HOST_FILE_PATH"

# Disable XFLAGS megatron mock module
XFLAGS --disable megatron

run_cmd="PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
         $OPTIONS_NCCL \
         torchrun $DISTRIBUTED_ARGS \
         $ROOT_DIR/../pretrain_chatglm.py \
         $GLM_ARGS \
         $TRAINING_ARGS \
         $OPTIMIZER_ARGS \
         $OUTPUT_ARGS "

echo ${run_cmd}
eval ${run_cmd}
