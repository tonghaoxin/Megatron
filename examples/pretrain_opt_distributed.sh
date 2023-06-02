#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=/datasets/data/opt/preprocess/part-00001-b9dcc04b-31ce-495a-a233-b309de593458-c000_text_document
CHECKPOINT_PATH=/checkpoint/opt6_7B
VOCAB_FILE=/checkpoint/opt6_7B_HF/vocab.json
MERGE_FILE=/checkpoint/opt6_7B_HF/merges.txt

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-12345}
NNODES=${1:-1}
NODE_RANK=${2:-0}

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

#--recompute-method uniform \
#--finetune
#--recompute-granularity full
#--distribute-saved-activations \

COMMON_ARGS="--clip-grad 1.0 \
             --weight-decay 1e-2 \
             --optimizer adam \
             --adam-beta1 0.9 \
             --adam-beta2 0.95 \
             --adam-eps 1e-08 \
             --train-iters 500000 \
             --lr-decay-iters 320000 \
             --lr-decay-style linear \
             --lr-warmup-fraction .01 \
             --no-async-tensor-model-parallel-allreduce \
             --fp16 \
             --tokenizer-type OptTokenizer \
             --use-learned-position-embeddings \
             --attention-dropout 0.1 \
             --hidden-dropout 0.1 \
             --embedding-dropout 0 \
             --use-flash-attn \
             --relu \
             --use-distributed-optimizer \
             --vocab-file $VOCAB_FILE \
             --merge-file $MERGE_FILE"

OPT_6_7B_ARGS="--tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --num-layers 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 16384 \
               --num-attention-heads 32 \
               --micro-batch-size 1 \
               --global-batch-size 32 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --lr 0.00012 \
               --min-lr 1.2e-5 "

OPT_66B_ARGS="--tensor-model-parallel-size 4 \
               --pipeline-model-parallel-size 2 \
               --num-layers 64 \
               --hidden-size 9216 \
               --ffn-hidden-size 36864 \
               --num-attention-heads 72 \
               --micro-batch-size 1 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --lr 8e-05 \
               --min-lr 8e-06"

OPT_175B_ARGS="--tensor-model-parallel-size 32 \
               --pipeline-model-parallel-size 32 \
               --num-layers 96 \
               --hidden-size 12288 \
               --ffn-hidden-size 49152 \
               --num-attention-heads 96 \
               --micro-batch-size 8 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --lr 3e-05 \
               --min-lr 3e-06"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"
OTHER_ARGS="--save $CHECKPOINT_PATH \
            --load $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"

PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_opt.py \
       $OPT_6_7B_ARGS \
       $COMMON_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
