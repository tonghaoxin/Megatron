#! /bin/bash

NAME="ChatGLM2-6B"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=/root/data/ChatGLM2-6B/ptuning/AdvertiseGen
CHECKPOINT_PATH=/workspace/THUDM/chatglm-6b/tp2-pp4-dp1

TENSORBOARD_PATH=./tensorborads/$EXP_NAME
LOG_PATH=./logs/$EXP_NAME

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9100}
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
FFN_HIDDEN_SIZE=13696

GLM_ARGS="--num-layers $NLAYERS \
          --hidden-size $HIDDEN_SIZE \
          --ffn-hidden-size $FFN_HIDDEN_SIZE \
          --num-attention-heads $NHEADS \
          --activation-func swiglu \
          --disable-bias-linear  \
          --add-qkv-bias true \
          --attention-softmax-in-fp32 \
          --rotary-embedding-2d-to-1d \
          --vocab-size 65024 \
	        --pad-vocab-size-to 65024 \
          --model-type-custom chatglm2-6b \
          --use-rotary-position-embeddings-with-position-ids \
          --seq2seq-data-collator "

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32

TP_SIZE=1
PP_SIZE=1
#SEQ_LEN=32768
SEQ_LEN=129

OPTIMIZER_ARGS="--optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.999 \
                --adam-eps 1e-8 \
                --lr 1e-4 \
                --min-lr 1e-6 \
                --override-opt_param-scheduler \
                --lr-decay-style cosine \
                --clip-grad 1.0 \
                --weight-decay 1e-1"

OUTPUT_ARGS="--log-interval 10 \
             --eval-interval 1000 \
             --eval-iters 3 \
             --tensorboard-dir $TENSORBOARD_PATH \
             --tensorboard-queue-size 5 \
             --log-timers-to-tensorboard \
             --log-batch-size-to-tensorboard \
             --log-validation-ppl-to-tensorboard \
             --load $CHECKPOINT_PATH \
             --save-interval 3000 \
             --save $CHECKPOINT_PATH"

TRAINING_ARGS="--train-iters 5000 \
               --lr-decay-iters 3200 \
               --lr-warmup-fraction 0.01 \
               --tensor-model-parallel-size $TP_SIZE \
               --pipeline-model-parallel-size $PP_SIZE \
               --micro-batch-size $MICRO_BATCH_SIZE \
               --global-batch-size $GLOBAL_BATCH_SIZE \
               --seq-length $SEQ_LEN \
               --num-workers 1 \
               --use-distributed-optimizer \
               --data-path $DATA_PATH \
               --split 949,50,1 \
               --distributed-backend nccl \
               --no-async-tensor-model-parallel-allreduce \
               --no-position-embedding \
               --tokenizer-model /root/data/ChatGLM2-6B/ptuning/THUDM/chatglm2-6b/tokenizer.model \
               --tokenizer-type ChatGLM2Tokenizer \
               --initial-loss-scale 65536 \
               --loss-scale 0 \
               --hidden-dropout 0 \
               --attention-dropout 0 \
               --embedding-dropout 0 \
               --init-method-std 1.0 \
               --multi-query-attention \
               --multi-query-group-num 2 \
	             --rmsnorm-epsilon 1e-05 \
	             --fp16-lm-cross-entropy \
               --fp16 \
               --no-bias-dropout-fusion \
               --dataloader-type cyclic \
               --preprocessing-num-workers 10 \
	             --untie-embeddings-and-output-weights"

OPTIONS_NCCL="NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_DISABLE=0"

#DISTRIBUTED_ARGS="--hostfile $HOST_FILE_PATH"

run_cmd="PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/cuda_ops':$PYTHONPATH \
         $OPTIONS_NCCL \
         torchrun $DISTRIBUTED_ARGS \
         $ROOT_DIR/../pretrain_chatglm2.py  \
         $GLM_ARGS \
         $TRAINING_ARGS \
         $OPTIMIZER_ARGS \
         $OUTPUT_ARGS "

echo ${run_cmd}
eval ${run_cmd}
