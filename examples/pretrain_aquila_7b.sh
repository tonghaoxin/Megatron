#! /bin/bash

# 7B 模型 BOS 地址
# bos://cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/aquilaChat-7b-tp8-pp1-dp1-fp32
# tokenizer and data
# bos://cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/aquilachat-7b

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

HF_TOKENIZER_PATH=$ROOT_DIR/../aquilachat
DATA_PATH=$HF_TOKENIZER_PATH/convo_samples.jsonl
CHECKPOINT_LOAD_PATH=$ROOT_DIR/../checkpoints_to_load
CHECKPOINT_SAVE_PATH=$ROOT_DIR/../checkpoints_to_save
TENSORBOARD_PATH=$ROOT_DIR/../tensorborads/$EXP_NAME

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-23461}
NNODES=${1:-1}
NODE_RANK=${2:-0}

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_MAX_CONNECTIONS=1


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

AQUILA_7B_ARGS="--tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 1 \
               --num-layers 32 \
               --num-attention-heads 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --micro-batch-size 2 \
               --global-batch-size 128 \
               --num-workers 1 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --pad-vocab-size-to 100008 \
               --lr 1e-5 \
               --min-lr 1.0e-5 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.98 \
               --weight-decay 0.1 \
               --adam-eps 1e-06 \
               --train-iters 100 \
               --lr-decay-iters 80 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .01 \
               --make-vocab-size-divisible-by 8 \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-type AQUILABPETokenizer \
               --rmsnorm-epsilon 1e-5 \
               --no-position-embedding \
               --no-gradient-accumulation-fusion \
               --no-bias-dropout-fusion \
               --no-bias-gelu-fusion \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --hf-tokenizer-path ${HF_TOKENIZER_PATH} \
               --loss-scale 0 \
               --initial-loss-scale 16 \
               --attention-softmax-in-fp32 \
               --untie-embeddings-and-output-weights \
               --swiglu \
               --sequence-parallel \
               --freqs-type complex \
               --no-masked-softmax-fusion \
               --use-rotary-position-embeddings \
               --no-query-key-layer-scaling \
               --dataloader-type cyclic"
               # --tensorboard-dir $TENSORBOARD_PATH \
               # --tensorboard-queue-size 5 \
               # --log-timers-to-tensorboard \
               # --log-batch-size-to-tensorboard \
               # --log-validation-ppl-to-tensorboard"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 2000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--load $CHECKPOINT_LOAD_PATH \
            --save $CHECKPOINT_SAVE_PATH \
            --data-impl mmap \
            --data-path $DATA_PATH \
            --split 100,0,0 \
            --distributed-backend nccl"


PYTHONPATH="$ROOT_DIR/../megatron/data":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_aquila.py \
       $AQUILA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
