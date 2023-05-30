#! /bin/bash

NAME="GLM-10B"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=/datasets/cc_news/cc_news_train_text_document
MULTITASK_DATA_PATH=""

TOKENIZER_MODEL=$ROOT_DIR/../megatron/chinese_sentencepiece/cog-pretrain.model
CHECKPOINT_PATH=$ROOT_DIR/../glm-10b-chinese/0531-tp1-pp1-dp8
TENSORBOARD_PATH=$ROOT_DIR/../tensorborads/$EXP_NAME
LOG_PATH=./logs/$EXP_NAME

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9981}
NNODES=${1:-1}
NODE_RANK=${2:-0}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

# Model args
# GLM 10B
NLAYERS=48
NHEADS=64
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=$((HIDDEN_SIZE*4))

GLM_ARGS="--num-layers $NLAYERS \
          --hidden-size $HIDDEN_SIZE \
          --ffn-hidden-size $FFN_HIDDEN_SIZE \
          --num-attention-heads $NHEADS \
          --make-vocab-size-divisible-by 64 \
          --block-lm \
          --gpt-prob 0.7 \
          --single-span-prob 0.02 \
          --short-seq-prob 0.02 \
          --mask-prob 0.15 \
          --average-block-length 3 \
          --min-gmask-ratio 0.2 \
          --aggregated-samples-per-sequence 4 \
          --task-mask \
          --bert-prob 0.5 \
          --gap-sentence-prob 0.3 \
          --gap-sentence-ratio 0.15 \
          --avg-block-length 3 \
          --gpt-min-ratio 0.25 \
          --block-mask-prob 0.1 \
          --short-seq-prob 0.02 \
          --no-bias-gelu-fusion"


MIXED_PRECISION_ARGS="--fp16 \
             --initial-loss-scale 16 \
             --min-loss-scale 1 \
             --loss-scale-window 1000 \
             --loss-scale 0 \
             --hysteresis 2"

# Training args
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=64
 #
TP_SIZE=1
PP_SIZE=1
SEQ_LEN=512
LENGTH_PER_SAMPLE=2000 # sequence length per sample from BinaryDataset

TRAIN_TOKENS=450000000000 # 450B tokens
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))  # Decay for the first 90% tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 5 / 1000))  # 0.5% warmup
BATCH_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 25 / 1000))  # 2.5% warmup

OPTIMIZER_ARGS="--optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.999 \
                --adam-eps 1e-8 \
                --lr 1e-5 \
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
               --train-samples $TRAIN_SAMPLES \
               --length-per-sample $LENGTH_PER_SAMPLE \
               --seq-length $SEQ_LEN \
               --max-position-embeddings 1025 \
               --num-workers 1 \
               --data-path $DATA_PATH \
               --load $CHECKPOINT_PATH \
               --skip-train-iteration-range 40701-40900 42401-42600 \
               --abort-on-unmet-fused-kernel-constraints \
               --split 949,50,1 \
               --distributed-backend nccl \
               --init-method-std 0.0052 \
               --hidden-dropout 0 \
               --attention-dropout 0 \
               --embedding-dropout 0 \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-type GLMChineseTokenizer \
               --tokenizer-model $TOKENIZER_MODEL \
               --use-distributed-optimizer \
               --no-query-key-layer-scaling \
               --model-type-custom glm-10b \
               --data-name cc-news \
               --dataloader-type cyclic \
               --fp16 "


PYTHONPATH='$ROOT_DIR/..':$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_glm_10b.py \
       $GLM_ARGS \
       $TRAINING_ARGS \
       $OPTIMIZER_ARGS \
       $MIXED_PRECISION_ARGS \
       $OUTPUT_ARGS
