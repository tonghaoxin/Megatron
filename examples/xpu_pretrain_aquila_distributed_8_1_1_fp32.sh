#! /bin/bash

'''
# 测试数据集 BOS 地址
bos:/klx-pytorch-work-bd/training/xpytorch/baige_megatron/aquila/data/
# aquila_7b tokenizer model BOS 地址
bos:/klx-pytorch-work-bd/training/xpytorch/baige_megatron/aquila/tokenizer_model/
# 7B 模型 BOS 地址
bos:/klx-pytorch-work-bd/training/xpytorch/baige_megatron/aquila/checkpoints/megatron_aquila_7b_checkpoint_tp8_pp1_fp32/
'''

set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

#DATA_PATH=${ROOT_DIR}/../data/llama_00_text_document
#DATA_PATH=${ROOT_DIR}/../megatron/data/my-aquila_conversations_document
DATA_PATH=/megatron_datasets/aquila/data/convo_samples.jsonl
CHECKPOINT_LOAD_PATH=/megatron_datasets/aquila/checkpoints/megatron_aquila_7b_checkpoint_tp8_pp1_fp32
CHECKPOINT_SAVE_PATH=/megatron_datasets/aquila/checkpoints/megatron_aquila_7b_checkpoint_tp8_pp1_fp32_save
TENSORBOARD_PATH='./aquila7b_tp8_pp1_fp32_tensorboard'
#VOCAB_FILE=${ROOT_DIR}/../megatron/data/checkpoints_in/aquilachat-7b/vocab.json
#MERGE_FILE=${ROOT_DIR}/../megatron/data/checkpoints_in/aquilachat-7b/merges.txt
HF_TOKENIZER_PATH=/megatron_datasets/aquila/tokenizer_model

export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export XPU_VISIBLE_DEVICES="4,5,6,7"
GPUS_PER_NODE=`echo "$XPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`

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

#fc autotune
export XPU_FC_AUTOTUNE_FILE="fc_autotune_aquila.log"
#fc use int16
export XDNN_FC_GEMM_DTYPE="int16"

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
               --global-batch-size 16 \
               --num-workers 1 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --lr 1e-5 \
               --min-lr 1.0e-5 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.98 \
               --weight-decay 0.1 \
               --adam-eps 1e-06 \
               --train-iters 10 \
               --lr-decay-iters 80 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .01 \
               --pad-vocab-size-to 100008 \
               --make-vocab-size-divisible-by 8 \
               --memory-saving \
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
               --tensorboard-queue-size 5 \
               --log-timers-to-tensorboard \
               --log-batch-size-to-tensorboard \
               --sequence-parallel \
               --swiglu \
               --freqs-type complex \
               --use-rotary-position-embeddings \
               --dataloader-type cyclic \
               --no-query-key-layer-scaling \
               --log-validation-ppl-to-tensorboard"
               #--no-masked-softmax-fusion \
               #--tensorboard-dir $TENSORBOARD_PATH \
               #--recompute-granularity selective \
               #--tokenizer-model $ROOT_DIR/../download/tokenizer.model \
               #--fp16 \
               #--use-flash-attn \
               #--fused-rmsnorm \
               # --vocab-file $VOCAB_FILE \
               # --merge-file $MERGE_FILE \


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
             --save-interval 100 "
             #--eval-interval 1000 \
             #--eval-iters 10"

OTHER_ARGS="--load $CHECKPOINT_LOAD_PATH \
            --data-impl mmap \
            --split 100,0,0 \
            --findmax-opt \
            --data-path $DATA_PATH \
            --exit-interval 2 \
            --no-init-megatron-time \
            --distributed-backend nccl"
            #--vocab-file $VOCAB_FILE \
            #--merge-file $MERGE_FILE \
            #--save $CHECKPOINT_SAVE_PATH \



ulimit -c 0
export XMLIR_F_XPU_ENABLED_BOOL=true
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=1800
#export BKCL_SOCKET_IFNAME=bond0

# debug
#export BKCL_DEBUG=1
#export XLOG_LEVEL="dist=info"


# use pcie
#export BKCL_PCIE_RING=1

# use ccix
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export ALLREDUCE_WORKER_SIZE=${GPUS_PER_NODE}
export BKCL_FORCE_SYNC=1

# bckl not use L3
#export BKCL_CCIX_BUFFER_GM=1
#export BKCL_RING_BUFFER_SIZE=524288

# xmlir set L3 size, default 13M
#export XMLIR_D_XPU_L3_SIZE=1048576

# Disable XFLAGS megatron mock module
# XFLAGS --disable megatron

PYTHONPATH=${ROOT_DIR}/../megatron/data
PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_aquila.py \
       $AQUILA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
