#! /bin/bash

NAME="GLM-130B"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=/ssd1/datasets/cc_news/cc_news_train_text_document
MULTITASK_DATA_PATH=""
CHECKPOINT_PATH=$ROOT_DIR/../checkpoint/glm-130b-md-tp4
TENSORBOARD_PATH=$ROOT_DIR/../tensorborads/$EXP_NAME
LOG_PATH=./logs/$EXP_NAME

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9988}
NNODES=${1:-1}
NODE_RANK=${2:-0}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

# Model args
# GLM 2B
NLAYERS=36
NHEADS=32
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=$((HIDDEN_SIZE * 8 / 3 / 2 * 2))
#FFN_HIDDEN_SIZE=$((HIDDEN_SIZE*4))

# GLM 10B
# NLAYERS=48
# NHEADS=64
# HIDDEN_SIZE=4096

# GLM 130B
#NLAYERS=70
#NHEADS=96
#HIDDEN_SIZE=12288
#FFN_HIDDEN_SIZE=$((HIDDEN_SIZE * 8 / 3))

GLM_ARGS="--num-layers $NLAYERS \
          --hidden-size $HIDDEN_SIZE \
          --ffn-hidden-size $FFN_HIDDEN_SIZE \
          --num-attention-heads $NHEADS \
          --make-vocab-size-divisible-by 768 \
          --glm \
          --gpt-prob 0.7 \
          --single-span-prob 0.02 \
          --short-seq-prob 0.02 \
          --mask-prob 0.15 \
          --average-block-length 3 \
          --min-gmask-ratio 0.2 \
          --aggregated-samples-per-sequence 4 \
          --use-rotary-position-embeddings \
          --geglu \
          --deepnorm \
          --no-bias-gelu-fusion"

# Training args
MICRO_BATCH_SIZE=1
# GLOBAL_BATCH_SIZE=4224 # 176 * 24
GLOBAL_BATCH_SIZE=352
#GLOBAL_BATCH_SIZE=528

TP_SIZE=2
PP_SIZE=2
SEQ_LEN=2048
LENGTH_PER_SAMPLE=2000 # sequence length per sample from BinaryDataset

TRAIN_TOKENS=450000000000 # 450B tokens
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))  # Decay for the first 90% tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 5 / 1000))  # 0.5% warmup
BATCH_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 25 / 1000))  # 2.5% warmup

OPTIMIZER_ARGS="--optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.95 \
                --adam-eps 1e-8 \
                --lr 4e-5 \
                --min-lr 4e-6 \
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

TRAINING_ARGS="--tensor-model-parallel-size $TP_SIZE \
               --pipeline-model-parallel-size $PP_SIZE \
               --micro-batch-size $MICRO_BATCH_SIZE \
               --global-batch-size $GLOBAL_BATCH_SIZE \
               --rampup-batch-size 2 2 4 \
               --train-samples $TRAIN_SAMPLES \
               --length-per-sample $LENGTH_PER_SAMPLE \
               --seq-length $SEQ_LEN \
               --num-workers 1 \
               --data-path $DATA_PATH \
               --skip-train-iteration-range 40701-40900 42401-42600 \
               --save $CHECKPOINT_PATH \
               --abort-on-unmet-fused-kernel-constraints \
               --split 949,50,1 \
               --distributed-backend nccl \
               --init-method-std 0.0052 \
               --shrink-logit-embedding-gradient \
               --shrink-embedding-gradient-alpha 0.1 \
               --no-position-embedding \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-model IceTokenizer \
               --use-distributed-optimizer \
               --recompute-granularity hybrid \
               --fp16"
               # --no-scatter-gather-tensors-in-pipeline \
               # --no-gradient-accumulation-fusion \
               # --load $CHECKPOINT_PATH \
               # --hidden-dropout 0 \
               # --recompute-activations \
               # --pipeline_model_parallel_split_rank \
               # --use-flash-attn \
	             # --multitask-data-path $MULTITASK_DATA_PATH \
	             # --multitask-ratio 0.05 \
               # --rampup-batch-size 192 24 $BATCH_WARMUP_SAMPLES \

OPTIONS_NCCL="NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_DISABLE=0"

run_cmd="PYTHONPATH='$ROOT_DIR/..':$PYTHONPATH \
         $OPTIONS_NCCL \
         python -m torch.distributed.launch $DISTRIBUTED_ARGS \
         $ROOT_DIR/../pretrain_glm.py \
         $GLM_ARGS \
         $TRAINING_ARGS \
         $OPTIMIZER_ARGS \
         $OUTPUT_ARGS "

mkdir -p $LOG_PATH
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $LOG_PATH/output.log
