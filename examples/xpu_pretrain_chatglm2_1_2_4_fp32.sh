#! /bin/bash

NAME="ChatGLM2-6B"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=/megatron_datasets/chatglm2/data/AdvertiseGen
TOKENIZER_PATH=/megatron_datasets/chatglm2/tokenizer_model/tokenizer.model
CHECKPOINT_PATH=/megatron_datasets/chatglm2/checkpoints/megatron_chatglm2_6b_checkpoint_tp2_pp4_fp32/

TENSORBOARD_PATH=./tensorborads/$EXP_NAME
LOG_PATH=./logs/$EXP_NAME

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9989}
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

TP_SIZE=2
PP_SIZE=4
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

OUTPUT_ARGS="--log-interval 1 \
             --eval-interval 1000 \
             --eval-iters 3 \
             --tensorboard-dir $TENSORBOARD_PATH \
             --tensorboard-queue-size 5 \
             --log-timers-to-tensorboard \
             --log-batch-size-to-tensorboard \
             --log-validation-ppl-to-tensorboard \
             --load $CHECKPOINT_PATH \
             --save-interval 3000 \
             "

#               --save $CHECKPOINT_PATH \
#               --load $CHECKPOINT_PATH \
TRAINING_ARGS="--train-iters 5 \
               --lr-decay-iters 3200 \
               --lr-warmup-fraction 0.01 \
               --tensor-model-parallel-size $TP_SIZE \
               --pipeline-model-parallel-size $PP_SIZE \
               --micro-batch-size $MICRO_BATCH_SIZE \
               --global-batch-size $GLOBAL_BATCH_SIZE \
               --seq-length $SEQ_LEN \
               --preprocessing-num-workers 1 \
               --data-path $DATA_PATH \
               --split 949,50,1 \
               --distributed-backend nccl \
               --no-async-tensor-model-parallel-allreduce \
               --no-position-embedding \
               --no-gradient-accumulation-fusion\
               --tokenizer-model $TOKENIZER_PATH \
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
               --no-bias-dropout-fusion \
               --dataloader-type cyclic \
               --memory-saving \
               --untie-embeddings-and-output-weights"
#   --fp16-lm-cross-entropy \
#                --fp16 \--use-distributed-optimizer \--no-contiguous-buffers-in-local-ddp\
ulimit -c 0
export XMLIR_F_XPU_ENABLED_BOOL=true
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=1800
export BKCL_FORCE_SYNC=1
# export BKCL_SOCKET_IFNAME=xgbe0

# debug
# export BKCL_DEBUG=1
# export XLOG_LEVEL="dist=info"
# export BKCL_CCIX_P2P_DEBUG=1

# use pcie
#export BKCL_PCIE_RING=1

# use ccix
export BKCL_CCIX_RING=1
# export BKCL_TREE_THRESHOLD=1
# export ALLREDUCE_WORKER_SIZE=${GPUS_PER_NODE}

# bckl not use L3
export BKCL_CCIX_BUFFER_GM=1
#export BKCL_RING_BUFFER_SIZE=524288

# xmlir set L3 size, default 13M
#export XMLIR_D_XPU_L3_SIZE=1048576

# python -m xacc.install

# Disable XFLAGS megatron mock module
XFLAGS --disable megatron

PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_chatglm2.py \
       $GLM_ARGS \
       $TRAINING_ARGS \
       $OPTIMIZER_ARGS \
       $OUTPUT_ARGS
