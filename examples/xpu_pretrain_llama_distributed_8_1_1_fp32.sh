#! /bin/bash

'''
# 测试数据集 BOS 地址
bos:/klx-pytorch-work-bd/training/qinduohao_llama/install/data/
# llama tokenizer model BOS 地址
bos:/klx-pytorch-work-bd/training/qinduohao_llama/install/tokenizer_model/
# 7B 模型 BOS 地址
bos:/klx-pytorch-work-bd/training/qinduohao_llama/install/checkpoints/megatron_llama_7b_checkpoint_tp8_pp1_fp32/
'''

set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

DATA_PATH=/megatron_datasets/llama/data/llama_00_text_document
CHECKPOINT_LOAD_PATH=/megatron_datasets/llama/checkpoints/megatron_llama_7b_checkpoint_tp8_pp1_fp32/
TOKENIZER_MODEL_PATH=/megatron_datasets/llama/tokenizer_model/tokenizer.model

#DATA_PATH=${ROOT_DIR}/../../data/llama_00_text_document
#CHECKPOINT_LOAD_PATH=${ROOT_DIR}/../../checkpoints/megatron_llama_7b_checkpoint_tp8_pp1_fp32
#CHECKPOINT_SAVE_PATH=${ROOT_DIR}/../../checkpoints/megatron_llama_7b_checkpoint_tp8_pp1_fp32_save
#TOKENIZER_MODEL_PATH=${ROOT_DIR}/../../download/tokenizer.model

TENSORBOARD_PATH='./llama7b_tp8_pp1_fp32_tensorboard'
# VOCAB_FILE=${ROOT_DIR}/../data/gpt2-vocab.json
# MERGE_FILE=${ROOT_DIR}/../data/gpt2-merges.txt


export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$XPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`

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

LLAMA_7B_ARGS="--tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 1 \
               --num-layers 32 \
               --exit-interval 2 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 2 \
               --global-batch-size 128 \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --lr 1.0e-5 \
               --min-lr 1.0e-5 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --weight-decay 0.1 \
               --adam-eps 1e-08 \
               --train-iters 500000 \
               --lr-decay-iters 320000 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .01 \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-type LLaMASentencePieceTokenizer \
               --tokenizer-model ${TOKENIZER_MODEL_PATH} \
               --activation-func swiglu \
               --use-rotary-position-embeddings \
               --rmsnorm-epsilon 1e-6 \
               --no-position-embedding \
               --no-gradient-accumulation-fusion \
               --no-bias-gelu-fusion \
               --no-masked-softmax-fusion \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --initial-loss-scale 16 \
               --untie-embeddings-and-output-weights"
               #--tensorboard-dir $TENSORBOARD_PATH \
               #--tensorboard-queue-size 5 \
               #--log-timers-to-tensorboard \
               #--log-batch-size-to-tensorboard \
               #--log-validation-ppl-to-tensorboard"
               #--fp16 \
               #--use-flash-attn \
               #--fused-rmsnorm \
               # --sequence-parallel
               # --vocab-file $VOCAB_FILE \
               # --merge-file $MERGE_FILE \

LLAMA_13B_ARGS="--tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 4 \
                --num-layers 40 \
                --hidden-size 5120 \
                --ffn-hidden-size 13824 \
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
                --tokenizer-type LLaMASentencePieceTokenizer \
                --tokenizer-model ${TOKENIZER_MODEL_PATH} \
                --activation-func swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-6 \
                --no-position-embedding \
                --no-gradient-accumulation-fusion \
                --no-bias-gelu-fusion \
                --disable-bias-linear \
                --attention-dropout 0 \
                --hidden-dropout 0"
                #--fp16 \
                #--use-flash-attn \
                #--fused-rmsnorm \
                # --sequence-parallel

LLAMA_33B_ARGS="--tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 4 \
                --num-layers 60 \
                --hidden-size 6656 \
                --ffn-hidden-size 17920 \
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
                --tokenizer-type LLaMASentencePieceTokenizer \
                --tokenizer-model ${TOKENIZER_MODEL_PATH} \
                --activation-func swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-6 \
                --no-position-embedding \
                --no-gradient-accumulation-fusion \
                --no-bias-gelu-fusion \
                --disable-bias-linear \
                --attention-dropout 0 \
                --hidden-dropout 0"
                #--fp16 \
                #--use-flash-attn \
                #--fused-rmsnorm \
                # --sequence-parallel

LLAMA_65B_ARGS="--tensor-model-parallel-size 4 \
                --pipeline-model-parallel-size 8 \
                --num-layers 80 \
                --hidden-size 8192 \
                --ffn-hidden-size 22016 \
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
                --tokenizer-type LLaMASentencePieceTokenizer \
                --tokenizer-model ${TOKENIZER_MODEL_PATH} \
                --swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-6 \
                --no-position-embedding \
                --no-gradient-accumulation-fusion \
                --no-bias-gelu-fusion \
                --disable-bias-linear \
                --attention-dropout 0 \
                --hidden-dropout 0"
                #--fp16 \
                #--use-flash-attn \
                # --sequence-parallel
                # --bf16
                # --fused-rmsnorm \

#OUTPUT_ARGS="--log-interval 10 \
#             --save-interval 10000 \
#             --eval-interval 1000 \
#             --eval-iters 10"

#OTHER_ARGS="--save $CHECKPOINT_PATH \
#            --load $CHECKPOINT_PATH \
#            --data-path $DATA_PATH \
#            --data-impl mmap \
#            --split 949,50,1 \
#            --distributed-backend nccl"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 500 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--load $CHECKPOINT_LOAD_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"
            #--save $CHECKPOINT_SAVE_PATH \

ulimit -c 0
export XMLIR_F_XPU_ENABLED_BOOL=true
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=1800
#export BKCL_SOCKET_IFNAME=xgbe0

# debug
#export BKCL_DEBUG=1
#export XLOG_LEVEL="dist=info"


# use pcie
#export BKCL_PCIE_RING=1

# use ccix
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1

# bckl not use L3
export BKCL_CCIX_BUFFER_GM=1
#export BKCL_RING_BUFFER_SIZE=524288

# xmlir set L3 size, default 13M
#export XMLIR_D_XPU_L3_SIZE=1048576

# Disable XFLAGS megatron mock module
XFLAGS --disable megatron

PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
