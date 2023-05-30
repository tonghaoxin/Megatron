#! /bin/bash

NAME="ChatGLM-6B"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=/workspace/tmp/wudao/
HF_TOKENIZER_PATH=/workspace/chatglm/ChatGLM-6B/THUDM/chatglm-6b/
CHECKPOINT_PATH=/workspace/chatglm/ChatGLM-6B/THUDM/tp1-pp1-dp8
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
          --glm \
          --rotary-embedding-2d \
          --model-type-custom chatglm-6b \
          --hf-tokenizer-path $HF_TOKENIZER_PATH \
          --gpt-prob 0.7 \
          --single-span-prob 0.02 \
          --short-seq-prob 0.02 \
          --mask-prob 0.15 \
          --average-block-length 3 \
          --min-gmask-ratio 0.2 \
          --aggregated-samples-per-sequence 4 \
          --use-rotary-position-embeddings \
          --deepnorm"

MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=128

TP_SIZE=1
PP_SIZE=1
SEQ_LEN=1024
LENGTH_PER_SAMPLE=2000 # sequence length per sample from BinaryDataset

TRAIN_TOKENS=4500000
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))  # Decay for the first 90% tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 5 / 1000))  # 0.5% warmup
BATCH_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 25 / 1000))  # 2.5% warmup

OPTIMIZER_ARGS="--optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.995 \
                --adam-eps 1e-8 \
                --lr 1e-5 \
                --min-lr 1e-6 \
                --override-opt_param-scheduler \
                --lr-decay-style cosine \
                --lr-decay-samples $LR_DECAY_SAMPLES \
                --lr-warmup-samples $LR_WARMUP_SAMPLES \
                --clip-grad 1.0 \
                --weight-decay 1e-1"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 100000000 \
             --eval-interval 1000 \
             --eval-iters 3 \
             --tensorboard-dir $TENSORBOARD_PATH \
             --tensorboard-queue-size 5 \
             --log-timers-to-tensorboard \
             --log-batch-size-to-tensorboard \
             --log-validation-ppl-to-tensorboard"

               #--recompute-activations \
TRAINING_ARGS="--tensor-model-parallel-size $TP_SIZE \
               --pipeline-model-parallel-size $PP_SIZE \
               --train-samples $TRAIN_SAMPLES \
               --length-per-sample $LENGTH_PER_SAMPLE \
               --micro-batch-size $MICRO_BATCH_SIZE \
               --global-batch-size $GLOBAL_BATCH_SIZE \
               --seq-length $SEQ_LEN \
               --num-workers 1 \
               --load $CHECKPOINT_PATH \
               --use-distributed-optimizer \
               --data-path $DATA_PATH \
               --abort-on-unmet-fused-kernel-constraints \
               --split 949,50,1 \
               --distributed-backend nccl \
               --init-method-std 0.0052 \
               --no-async-tensor-model-parallel-allreduce \
               --no-gradient-accumulation-fusion \
               --shrink-embedding-gradient-alpha 1 \
               --no-position-embedding \
               --tokenizer-model /workspace/chatglm/model/ChatGLM-6B/THUDM/chatglm-6b/ice_text.model\
               --initial-loss-scale 16 \
               --min-loss-scale 1 \
               --loss-scale-window 1000 \
               --hysteresis 2 \
               --loss-scale 0 \
               --hidden-dropout 0 \
               --attention-dropout 0 \
               --embedding-dropout 0 \
               --fp16"

OPTIONS_NCCL="NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_DISABLE=0"

#DISTRIBUTED_ARGS="--hostfile $HOST_FILE_PATH"

run_cmd="PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/fused_kernels':$PYTHONPATH \
         $OPTIONS_NCCL \
         python -m torch.distributed.launch $DISTRIBUTED_ARGS \
         $ROOT_DIR/../pretrain_chatglm.py \
         $GLM_ARGS \
         $TRAINING_ARGS \
         $OPTIMIZER_ARGS \
         $OUTPUT_ARGS "

mkdir -p $LOG_PATH
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $LOG_PATH/output.log
