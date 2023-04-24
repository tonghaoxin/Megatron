#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../data/llama_00_text_document
CHECKPOINT_PATH=${ROOT_DIR}/../checkpoints/llama_65b
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

LLAMA_7B_ARGS="--tensor-model-parallel-size 2 \
               --pipeline-model-parallel-size 2 \
               --num-layers 32 \
               --hidden-size 4096 \
               --num-attention-heads 32 \
               --micro-batch-size 1 \
               --global-batch-size 512 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --lr 0.0003 \
               --min-lr 1.0e-5 \
               --clip-grad 1.0 \
               --weight-decay 1e-2 \
               --train-iters 500000 \
               --lr-decay-iters 320000 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .01 \
               --recompute-method uniform \
               --no-async-tensor-model-parallel-allreduce \
               --fp16
               --tokenizer-type SentencePieceTokenizer \
               --tokenizer-model $ROOT_DIR/../download/tokenizer.model \
               --swiglu \
               --use-rotary-position-embeddings \
               --rmsnorm-epsilon 1e-6 \
               --fused-rmsnorm \
               --no-position-embedding \
               --disable-bias-linear \
               --use-flash-attn"
               # --ffn-hidden-size 11008 \
               # --sequence-parallel
               # --vocab-file $VOCAB_FILE \
               # --merge-file $MERGE_FILE \

LLAMA_13B_ARGS="--tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 4 \
                --num-layers 40 \
                --hidden-size 5120 \
                --num-attention-heads 40 \
                --micro-batch-size 1 \
                --global-batch-size 512 \
                --seq-length 2048 \
                --max-position-embeddings 2048 \
                --lr 0.0003 \
                --min-lr 1.0e-5 \
                --clip-grad 1.0 \
                --weight-decay 1e-2 \
                --train-iters 500000 \
                --lr-decay-iters 320000 \
                --lr-decay-style cosine \
                --lr-warmup-fraction .01 \
                --recompute-method uniform \
                --no-async-tensor-model-parallel-allreduce \
                --fp16
                --tokenizer-type SentencePieceTokenizer \
                --tokenizer-model $ROOT_DIR/../download/tokenizer.model \
                --swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-6 \
                --fused-rmsnorm \
                --no-position-embedding \
                --disable-bias-linear \
                --use-flash-attn"
                # --sequence-parallel

LLAMA_33B_ARGS="--tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 4 \
                --num-layers 60 \
                --hidden-size 6656 \
                --num-attention-heads 52 \
                --micro-batch-size 1 \
                --global-batch-size 4096 \
                --seq-length 2048 \
                --max-position-embeddings 2048 \
                --lr 0.00015 \
                --min-lr 1.0e-5 \
                --clip-grad 1.0 \
                --weight-decay 1e-2 \
                --train-iters 500000 \
                --lr-decay-iters 320000 \
                --lr-decay-style cosine \
                --lr-warmup-fraction .01 \
                --recompute-method uniform \
                --no-async-tensor-model-parallel-allreduce \
                --fp16
                --tokenizer-type SentencePieceTokenizer \
                --tokenizer-model $ROOT_DIR/../download/tokenizer.model \
                --swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-6 \
                --fused-rmsnorm \
                --no-position-embedding \
                --disable-bias-linear \
                --use-flash-attn"
                # --sequence-parallel

LLAMA_65B_ARGS="--tensor-model-parallel-size 4 \
                --pipeline-model-parallel-size 8 \
                --num-layers 80 \
                --hidden-size 8192 \
                --num-attention-heads 64 \
                --micro-batch-size 1 \
                --global-batch-size 512 \
                --seq-length 2048 \
                --max-position-embeddings 2048 \
                --lr 0.00015 \
                --min-lr 1.0e-5 \
                --clip-grad 1.0 \
                --weight-decay 1e-2 \
                --train-iters 500000 \
                --lr-decay-iters 320000 \
                --lr-decay-style cosine \
                --lr-warmup-fraction .01 \
                --recompute-method uniform \
                --no-async-tensor-model-parallel-allreduce \
                --fp16
                --tokenizer-type SentencePieceTokenizer \
                --tokenizer-model $ROOT_DIR/../download/tokenizer.model \
                --swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-6 \
                --fused-rmsnorm \
                --no-position-embedding \
                --disable-bias-linear \
                --use-flash-attn"
                # --sequence-parallel
                # --bf16

OUTPUT_ARGS="--log-interval 10 \
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
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
