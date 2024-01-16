#! /bin/bash

'''
# 测试数据集 BOS 地址
bos:/klx-pytorch-work-bd/training/xpytorch/baige_megatron/bloom/data/
# aquila_7b tokenizer model BOS 地址
bos:/klx-pytorch-work-bd/training/xpytorch/baige_megatron/bloom/tokenizer_model/
# 7B 模型 BOS 地址
bos:/klx-pytorch-work-bd/training/xpytorch/baige_megatron/bloom/checkpoints/megatron_bloom_7b_checkpoint_tp8_pp1_fp32/
'''



NAME="bloom-7b"
TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${NAME}-${TIMESTAMP}

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_DIR=/megatron_datasets/bloom

DATA_PATH=$DATA_DIR/data/openwebtext-10k_text_document
CHECKPOINT_SAVE_PATH=$DATA_DIR/${NAME}_save
CHECKPOINT_LOAD_PATH=$DATA_DIR/checkpoints/megatron_bloom_7b_checkpoint_tp8_pp1_fp32
TOKENIZER_PATH=$DATA_DIR/tokenizer_model
TENSORBOARD_PATH=./logs/$EXP_NAME

export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$XPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}

# bckl not use L3
export BKCL_CCIX_BUFFER_GM=1
# solve 8-card lost accuracy
export  BKCL_TREE_THRESHOLD=1

export BKCL_FORCE_SYNC=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

BLOOM_7B_ARGS="--tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 1 \
               --num-layers 30 \
               --hidden-size 4096 \
               --num-attention-heads 32 \
               --micro-batch-size 4 \
               --global-batch-size 64 \
               --seq-length 512 \
               --max-position-embeddings 512 \
               --lr 2e-5 \
               --min-lr 1e-6 \
               --clip-grad 1.0 \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --weight-decay 1e-4 \
               --train-iters 1000 \
               --lr-decay-iters 320000 \
               --lr-decay-style cosine \
               --lr-warmup-fraction 0 \
               --tokenizer-type BloomHFTokenizer \
               --hf-tokenizer-path $TOKENIZER_PATH \
               --embedding-dropout 0 \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --use-alibi-position-embeddings \
               --activation-func openai_gelu \
               --pad-vocab-size-to 250880 \
               --layernorm-epsilon 1e-05 \
               --add-embed-layernorm \
               --no-async-tensor-model-parallel-allreduce \
               --no-contiguous-buffers-in-local-ddp \
               --no-gradient-accumulation-fusion \
               --no-load-optim \
               --no-load-rng \
               --no-masked-softmax-fusion \
               --no-position-embedding \
               --no-query-key-layer-scaling \
               --initial-loss-scale 16 \
               --memory-saving \
               --exit-interval 2 \
               "

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 100 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--load $CHECKPOINT_LOAD_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl \
            "

TENSORBOARD_ARGS="--tensorboard-dir $TENSORBOARD_PATH \
                  --tensorboard-queue-size 5 \
                  --log-timers-to-tensorboard \
                  --log-batch-size-to-tensorboard \
                  --log-validation-ppl-to-tensorboard"

# Disable XFLAGS megatron mock module
XFLAGS --disable megatron

PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_bloom.py \
       $BLOOM_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
