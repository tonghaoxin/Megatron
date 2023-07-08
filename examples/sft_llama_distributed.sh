#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../data/alpaca_data.json
CHECKPOINT_PATH=${ROOT_DIR}/../checkpoints/llama-7b
# VOCAB_FILE=${ROOT_DIR}/../data/gpt2-vocab.json
# MERGE_FILE=${ROOT_DIR}/../data/gpt2-merges.txt


GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA_7B_ARGS="--seed 42 \
               --tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --num-layers 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 4 \
               --global-batch-size 256 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --lr 5e-5 \
               --clip-grad 1.0 \
               --weight-decay 0. \
               --train-iters 609 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .03 \
               --recompute-method uniform \
               --no-async-tensor-model-parallel-allreduce \
               --fp16 \
               --tokenizer-type LLaMASFTTokenizerHF \
               --tokenizer-model $ROOT_DIR/../download/tokenizer.model \
               --activation-func swiglu \
               --use-rotary-position-embeddings \
               --rmsnorm-epsilon 1e-6 \
               --fused-rmsnorm \
               --no-position-embedding \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --use-flash-attn \
               --untie-embeddings-and-output-weights \
               --sft-collator \
               --dataloader-type cyclic"
               # --sequence-parallel
               # --vocab-file $VOCAB_FILE \
               # --merge-file $MERGE_FILE \

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 300 \
             --eval-interval 300 \
             --eval-iters 10"

mkdir -p $CHECKPOINT_PATH/tensorborad_log

OTHER_ARGS="--tensorboard-dir $CHECKPOINT_PATH/tensorborad_log \
            --save $CHECKPOINT_PATH \
            --load $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"

PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../sft_llama.py \
       $LLAMA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
