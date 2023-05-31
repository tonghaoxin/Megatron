#! /bin/bash

NAME="GLM-130B"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
# DATA_PATH=$ROOT_DIR/../data/llama_00_text_document
# DATA_PATH=/mnt/cfs/glm/train_data/train_data
#DATA_PATH=/mnt/cfs/glm/train_data/cc_news_train_text_document
DATA_PATH=/datasets/cc_news/cc_news_train_text_document
MULTITASK_DATA_PATH=""
CHECKPOINT_PATH=./checkpoint/glm-130b-md-tp4
#CHECKPOINT_PATH=/mnt/cfs/glm/checkpoints
TENSORBOARD_PATH=./tensorborads/$EXP_NAME
#LOG_PATH=/mnt/cfs/glm/logs/$EXP_NAME
LOG_PATH=./logs/$EXP_NAME
# VOCAB_FILE=${ROOT_DIR}/../data/gpt2-vocab.json
# MERGE_FILE=${ROOT_DIR}/../data/gpt2-merges.txt

#NUM_WORKERS=${1:-1}
#NUM_GPUS_PER_WORKER=8
#HOST_FILE_PATH=${HOST_FILE_PATH:-"/etc/mpi/hostfile"}
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
ENCODER_NUM_LAYERS=$((NLAYERS / 2))
DECODER_NUM_LAYERS=$((NLAYERS / 2))
          #--encoder-num-layers $NLAYERS \
          #--decoder-num-layers $NLAYERS \

GLM_ARGS="--num-layers $NLAYERS \
          --hidden-size $HIDDEN_SIZE \
          --ffn-hidden-size $FFN_HIDDEN_SIZE \
          --num-attention-heads $NHEADS \
          --make-vocab-size-divisible-by 768 \
          --gpt-prob 0.7 \
          --single-span-prob 0.02 \
          --short-seq-prob 0.02 \
          --mask-prob 0.15 \
          --average-block-length 3 \
          --min-gmask-ratio 0.2 \
          --aggregated-samples-per-sequence 4 \
          --use-rotary-position-embeddings \
          --use-rotary-position-embeddings-with-position-ids \
          --geglu \
          --no-bias-gelu-fusion"
          #--deepnorm \
#--num-layers $NLAYERS \

# Training args
MICRO_BATCH_SIZE=1
# GLOBAL_BATCH_SIZE=4224 # 176 * 24
# GLOBAL_BATCH_SIZE=352
GLOBAL_BATCH_SIZE=528

TP_SIZE=2
PP_SIZE=4
SEQ_LEN=2048
LENGTH_PER_SAMPLE=2000 # sequence length per sample from BinaryDataset

TRAIN_TOKENS=450000000000 # 450B tokens
TRAIN_SAMPLES=$((TRAIN_TOKENS / SEQ_LEN))
LR_DECAY_SAMPLES=$((TRAIN_SAMPLES * 90 / 100))  # Decay for the first 90% tokens then continue at fixed --min-lr
LR_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 5 / 1000))  # 0.5% warmup
BATCH_WARMUP_SAMPLES=$((TRAIN_SAMPLES * 25 / 1000))  # 2.5% warmup

ZERO_STAGE=1

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
               --rampup-batch-size 3 3 9 \
               --train-samples $TRAIN_SAMPLES \
               --length-per-sample $LENGTH_PER_SAMPLE \
               --seq-length $SEQ_LEN \
               --num-workers 1 \
               --data-path $DATA_PATH \
               --save $CHECKPOINT_PATH \
               --abort-on-unmet-fused-kernel-constraints \
               --split 949,50,1 \
               --distributed-backend nccl \
               --recompute-activations \
               --init-method-std 0.0052 \
               --tokenizer-type IceTokenizer\
               --no-async-tensor-model-parallel-allreduce \
               --no-gradient-accumulation-fusion \
               --shrink-word-embedding \
               --shrink-logit-embedding-gradient \
               --shrink-embedding-gradient-alpha 0.1 \
               --no-position-embedding \
               --fp16"
               # --no-gradient-accumulation-fusion \
               # --pipeline_model_parallel_split_rank
               # --skip-train-iteration-range 40701-40900 42401-42600 \
               # --load $CHECKPOINT_PATH \
               # --use-flash-attn \
               # --load-deepspeed-model-only \
	             # --multitask-data-path $MULTITASK_DATA_PATH \
	             # --multitask-ratio 0.05 \
               # --tokenizer-type IceTokenizer"
               # --rampup-batch-size 192 24 $BATCH_WARMUP_SAMPLES \
               # --vocab-file $VOCAB_FILE \
               # --merge-file $MERGE_FILE \
               # --tokenizer-type IceTokenizer \
               # --tokenizer-model $ROOT_DIR/../download/tokenizer.model \
               # --tokenizer-name-or-path $ROOT_DIR/../download/tokenizer.model \

OPTIONS_NCCL="NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8 NCCL_IB_DISABLE=0"

#DISTRIBUTED_ARGS="--hostfile $HOST_FILE_PATH"

run_cmd="PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/fused_kernels':$PYTHONPATH \
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
