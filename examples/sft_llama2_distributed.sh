#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../../data/alpaca_data_en_52k.json
#TOKENIZER_PATH=$ROOT_DIR/../../checkpoints/Llama-2-7b-hf/tokenizer.model
#CHECKPOINT_PATH=${ROOT_DIR}/../../checkpoints/Llama-2-7b-tp1-pp1-zero1
#TOKENIZER_PATH=$ROOT_DIR/../../checkpoints/Llama-2-13b-hf/tokenizer.model
#CHECKPOINT_PATH=${ROOT_DIR}/../../checkpoints/Llama-2-13b-tp2-pp1-dp4-zero1
TOKENIZER_PATH=$ROOT_DIR/../../checkpoints/Llama-2-70b-hf/tokenizer.model
CHECKPOINT_PATH=${ROOT_DIR}/../../checkpoints/Llama-2-70b-tp8-pp1-dp2-zero1


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

LLAMA2_7B_ARGS="--seed 42 \
                --tensor-model-parallel-size 1 \
                --pipeline-model-parallel-size 1 \
                --num-layers 32 \
                --hidden-size 4096 \
                --ffn-hidden-size 11008 \
                --num-attention-heads 32 \
                --micro-batch-size 1 \
                --global-batch-size 128 \
                --seq-length 2048 \
                --max-position-embeddings 2048 \
                --lr 2e-5 \
                --clip-grad 1.0 \
                --weight-decay 0. \
                --train-iters 1218 \
                --lr-decay-style cosine \
                --lr-warmup-fraction .03 \
                --recompute-method uniform \
                --no-async-tensor-model-parallel-allreduce \
                --fp16 \
                --tokenizer-type LLaMASFTTokenizerHF \
                --tokenizer-model $TOKENIZER_PATH \
                --activation-func swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-5 \
                --fused-rmsnorm \
                --no-position-embedding \
                --disable-bias-linear \
                --use-distributed-optimizer \
                --untie-embeddings-and-output-weights \
                --attention-dropout 0 \
                --hidden-dropout 0 \
                --embedding-dropout 0 \
                --sft-collator \
                --variable-seq-lengths \
                --use-flash-attn \
                --dataloader-type cyclic \
                --initial-loss-scale 16"
                # --variable-seq-lengths \
                # --use-flash-attn \
		# ....correspond...hf
                # --fused-rmsnorm \
                # ..................
                # --enable-debug \
                # --debug-tp-list 0 \
                # --debug-pp-list 0 \
                # --debug-dp-list 0 \
                # --debug-layer-list 25 26 27 28 29 30 31 32 \
                # --debug-save-path ${ROOT_DIR}/../debug_info \
                # --sequence-parallel
                #  padding issues
                # ..............
                # --attn-mask-pad \
                # --attention-softmax-in-fp32 \
                # ...............
                # ....close fusion and fused
                # --no-masked-softmax-fusion \
                # --no-bias-dropout-fusion \
                # --no-bias-gelu-fusion \
                # --no-gradient-accumulation-fusion \
                #..........................
                # --multi-query-attention \
                # --multi-query-group-num 32 \


LLAMA2_13B_ARGS="--seed 42 \
                 --tensor-model-parallel-size 2 \
                 --pipeline-model-parallel-size 1 \
                 --num-layers 40 \
                 --hidden-size 5120 \
                 --ffn-hidden-size 13824 \
                 --num-attention-heads 40 \
                 --micro-batch-size 1 \
                 --global-batch-size 512 \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --lr 2e-5 \
                 --clip-grad 1.0 \
                 --weight-decay 0. \
                 --train-iters 304 \
                 --lr-decay-style cosine \
                 --lr-warmup-fraction .03 \
                 --recompute-method uniform \
                 --no-async-tensor-model-parallel-allreduce \
                 --fp16 \
                 --tokenizer-type LLaMASFTTokenizerHF \
                 --tokenizer-model $TOKENIZER_PATH \
                 --activation-func swiglu \
                 --use-rotary-position-embeddings \
                 --rmsnorm-epsilon 1e-5 \
                 --fused-rmsnorm \
                 --no-position-embedding \
                 --disable-bias-linear \
                 --use-flash-attn \
                 --use-distributed-optimizer \
                 --untie-embeddings-and-output-weights \
                 --attention-dropout 0 \
                 --hidden-dropout 0 \
                 --embedding-dropout 0 \
                 --multi-query-attention \
                 --multi-query-group-num 40 \
                 --sft-collator \
                 --dataloader-type cyclic"
                 # --sequence-parallel


LLAMA2_70B_ARGS="--seed 42 \
                 --tensor-model-parallel-size 8 \
                 --pipeline-model-parallel-size 1 \
                 --num-layers 80 \
                 --hidden-size 8192 \
                 --ffn-hidden-size 28672 \
                 --num-attention-heads 64 \
                 --micro-batch-size 1 \
                 --global-batch-size 16 \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --lr 2e-5 \
                 --clip-grad 1.0 \
                 --weight-decay 0. \
                 --train-iters 9750 \
                 --lr-decay-style cosine \
                 --lr-warmup-fraction .03 \
                 --recompute-method uniform \
                 --no-async-tensor-model-parallel-allreduce \
                 --fp16 \
                 --tokenizer-type LLaMASFTTokenizerHF \
                 --tokenizer-model $TOKENIZER_PATH \
                 --activation-func swiglu \
                 --use-rotary-position-embeddings \
                 --rmsnorm-epsilon 1e-5 \
                 --fused-rmsnorm \
                 --no-position-embedding \
                 --disable-bias-linear \
                 --use-flash-attn \
                 --use-distributed-optimizer \
                 --untie-embeddings-and-output-weights \
                 --attention-dropout 0 \
                 --hidden-dropout 0 \
                 --embedding-dropout 0 \
                 --multi-query-attention \
                 --multi-query-group-num 8 \
                 --ffn-dim-multiplier 1.3 \
                 --multiple-of 4096 \
                 --sft-collator \
                 --variable-seq-lengths \
                 --dataloader-type cyclic"
                 # --sequence-parallel


OUTPUT_ARGS="--log-interval 1 \
             --save-interval 400 \
             --eval-interval 400 \
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
       $LLAMA2_70B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
