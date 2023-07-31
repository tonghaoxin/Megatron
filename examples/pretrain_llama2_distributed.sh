#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../data/llama_00_text_document
TOKENIZER_PATH=$ROOT_DIR/../tokenizer/tokenizer.model
CHECKPOINT_PATH=${ROOT_DIR}/../checkpoints/
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

LLAMA2_7B_ARGS="--tensor-model-parallel-size 1 \
                --pipeline-model-parallel-size 1 \
                --num-layers 32 \
                --hidden-size 4096 \
                --ffn-hidden-size 11008 \
                --num-attention-heads 32 \
                --micro-batch-size 1 \
                --global-batch-size 128 \
                --seq-length 4096 \
                --max-position-embeddings 4096 \
                --lr 0.0003 \
                --min-lr 1.0e-5 \
                --clip-grad 1.0 \
                --weight-decay 1e-2 \
                --optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.95 \
                --adam-eps 1e-05 \
                --train-iters 500000 \
                --lr-decay-iters 320000 \
                --lr-decay-style cosine \
                --lr-warmup-fraction .01 \
                --no-async-tensor-model-parallel-allreduce \
                --tokenizer-type LLaMASentencePieceTokenizer \
                --tokenizer-model $TOKENIZER_PATH \
                --activation-func swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-5 \
                --fused-rmsnorm \
                --no-position-embedding \
                --disable-bias-linear \
                --attention-dropout 0 \
                --hidden-dropout 0 \
                --embedding-dropout 0 \
                --use-distributed-optimizer \
                --untie-embeddings-and-output-weights \
                --use-flash-attn \
                --fp16"
                # --sequence-parallel
                # --initial-loss-scale 16 \
                # --multi-query-attention \
                # --multi-query-group-num 32 \

LLAMA2_13B_ARGS="--tensor-model-parallel-size 2 \
                 --pipeline-model-parallel-size 1 \
                 --num-layers 40 \
                 --hidden-size 5120 \
                 --ffn-hidden-size 13824 \
                 --num-attention-heads 40 \
                 --micro-batch-size 1 \
                 --global-batch-size 512 \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --lr 0.0003 \
                 --min-lr 1.0e-5 \
                 --clip-grad 1.0 \
                 --weight-decay 1e-2 \
                 --optimizer adam \
                 --adam-beta1 0.9 \
                 --adam-beta2 0.95 \
                 --adam-eps 1e-05 \
                 --train-iters 500000 \
                 --lr-decay-iters 320000 \
                 --lr-decay-style cosine \
                 --lr-warmup-fraction .01 \
                 --no-async-tensor-model-parallel-allreduce \
                 --tokenizer-type LLaMASentencePieceTokenizer \
                 --tokenizer-model $TOKENIZER_PATH \
                 --activation-func swiglu \
                 --use-rotary-position-embeddings \
                 --rmsnorm-epsilon 1e-5 \
                 --fused-rmsnorm \
                 --no-position-embedding \
                 --disable-bias-linear \
                 --attention-dropout 0 \
                 --hidden-dropout 0 \
                 --embedding-dropout 0 \
                 --use-distributed-optimizer \
                 --untie-embeddings-and-output-weights \
                 --use-flash-attn \
                 --multi-query-attention \
                 --multi-query-group-num 40 \
                 --fp16"
                 # --sequence-parallel

# LLAMA2_34B_ARGS="--tensor-model-parallel-size 2 \
#                  --pipeline-model-parallel-size 4 \
#                  --num-layers 60 \
#                  --hidden-size 6656 \
#                  --ffn-hidden-size 17920 \
#                  --num-attention-heads 52 \
#                  --micro-batch-size 1 \
#                  --global-batch-size 4096 \
#                  --seq-length 2048 \
#                  --max-position-embeddings 2048 \
#                  --lr 0.00015 \
#                  --min-lr 1.0e-5 \
#                  --clip-grad 1.0 \
#                  --weight-decay 1e-2 \
#                  --optimizer adam \
#                  --adam-beta1 0.9 \
#                  --adam-beta2 0.95 \
#                  --adam-eps 1e-05 \
#                  --train-iters 500000 \
#                  --lr-decay-iters 320000 \
#                  --lr-decay-style cosine \
#                  --lr-warmup-fraction .01 \
#                  --recompute-method uniform \
#                  --no-async-tensor-model-parallel-allreduce \
#                  --fp16
#                  --tokenizer-type LLaMASentencePieceTokenizer \
#                  --tokenizer-model $ROOT_DIR/../download/tokenizer.model \
#                  --activation-func swiglu \
#                  --use-rotary-position-embeddings \
#                  --rmsnorm-epsilon 1e-6 \
#                  --fused-rmsnorm \
#                  --no-position-embedding \
#                  --disable-bias-linear \
#                  --attention-dropout 0 \
#                  --hidden-dropout 0 \
#                  --embedding-dropout 0 \
#                  --untie-embeddings-and-output-weights \
#                  --use-flash-attn"
#                  # --sequence-parallel

LLAMA2_70B_ARGS="--tensor-model-parallel-size 8 \
                 --pipeline-model-parallel-size 1 \
                 --num-layers 80 \
                 --hidden-size 8192 \
                 --ffn-hidden-size 28672 \
                 --num-attention-heads 64 \
                 --micro-batch-size 1 \
                 --global-batch-size 512 \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --lr 0.00015 \
                 --min-lr 1.0e-5 \
                 --clip-grad 1.0 \
                 --weight-decay 1e-2 \
                 --optimizer adam \
                 --adam-beta1 0.9 \
                 --adam-beta2 0.95 \
                 --adam-eps 1e-05 \
                 --train-iters 500000 \
                 --lr-decay-iters 320000 \
                 --lr-decay-style cosine \
                 --lr-warmup-fraction .01 \
                 --recompute-method uniform \
                 --no-async-tensor-model-parallel-allreduce \
                 --tokenizer-type LLaMASentencePieceTokenizer \
                 --tokenizer-model $TOKENIZER_PATH \
                 --swiglu \
                 --use-rotary-position-embeddings \
                 --rmsnorm-epsilon 1e-5 \
                 --fused-rmsnorm \
                 --no-position-embedding \
                 --disable-bias-linear \
                 --attention-dropout 0 \
                 --hidden-dropout 0 \
                 --embedding-dropout 0 \
                 --use-distributed-optimizer \
                 --untie-embeddings-and-output-weights \
                 --use-flash-attn \
                 --multi-query-attention \
                 --multi-query-group-num 8 \
                 --ffn-dim-multiplier 1.3 \
                 --multiple-of 4096 \
                 --fp16"
                 # --sequence-parallel

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
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA2_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
