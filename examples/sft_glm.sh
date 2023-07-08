#! /bin/bash

NAME="GLM-10B"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../data/alpaca_data.json

TOKENIZER_MODEL=$ROOT_DIR/../download/glm-10b-tokenizers/cog-pretrain.model
CHECKPOINT_PATH=$ROOT_DIR/../checkpoints/glm-10b/
TENSORBOARD_PATH=$ROOT_DIR/../tensorborads/$EXP_NAME

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
          --make-vocab-size-divisible-by 128 \
          --block-lm \
          --task-mask"

# Training args
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=64
 #
TP_SIZE=1
PP_SIZE=1
SEQ_LEN=1024

OPTIMIZER_ARGS="--optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.95 \
                --adam-eps 1e-8 \
                --lr 5e-6 \
                --min-lr 6e-8 \
                --override-opt_param-scheduler \
                --lr-decay-style linear \
                --lr-decay-samples 520020 \
                --lr-warmup-samples 1 \
                --clip-grad 1.0 \
                --weight-decay 1e-2"

OUTPUT_ARGS="--log-interval 50 \
             --save-interval 1000 \
             --save $CHECKPOINT_PATH \
             --eval-interval 1000 \
             --eval-iters 100 \
             --tensorboard-dir $TENSORBOARD_PATH \
             --tensorboard-queue-size 5 \
             --log-timers-to-tensorboard \
             --log-batch-size-to-tensorboard \
             --log-validation-ppl-to-tensorboard"

TRAINING_ARGS="--tensor-model-parallel-size $TP_SIZE \
               --pipeline-model-parallel-size $PP_SIZE \
               --micro-batch-size $MICRO_BATCH_SIZE \
               --global-batch-size $GLOBAL_BATCH_SIZE \
               --seq-length $SEQ_LEN \
               --max-position-embeddings 1025 \
               --data-path $DATA_PATH \
               --load $CHECKPOINT_PATH \
               --abort-on-unmet-fused-kernel-constraints \
               --train-samples 520020 \
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
               --recompute-granularity full \
               --recompute-method block \
               --recompute-num-layers 8 \
               --initial-loss-scale 16 \
               --min-loss-scale 1 \
               --loss-scale-window 1000 \
               --loss-scale 0 \
               --hysteresis 2 \
               --fp16 \
               --label-smoothing 0.1 \
               --src-seq-length 512 \
               --tgt-seq-length 128 \
               --custom-collator"


PYTHONPATH='$ROOT_DIR/..':$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../sft_glm.py \
       $GLM_ARGS \
       $TRAINING_ARGS \
       $OPTIMIZER_ARGS \
       $OUTPUT_ARGS
