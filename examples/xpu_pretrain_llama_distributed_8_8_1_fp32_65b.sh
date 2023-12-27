#! /bin/bash

# 测试数据集 BOS 地址
# bos:/klx-pytorch-work-bd/training/qinduohao_llama/install/data/
# llama tokenizer model BOS 地址
# bos:/klx-pytorch-work-bd/training/qinduohao_llama/install/tokenizer_model/
# 7B 模型 BOS 地址
# bos:/klx-pytorch-work-bd/training/qinduohao_llama/install/checkpoints/megatron_llama_7b_checkpoint_tp8_pp1_fp32/

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../data/llama_00_text_document
TOKENIZER_MODEL_PATH=$ROOT_DIR/../data/tokenizer.model

export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$XPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

# bckl not use L3
export BKCL_CCIX_BUFFER_GM=1
# solve 8-card lost accuracy
export  BKCL_TREE_THRESHOLD=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"


LLAMA_65B_ARGS="--tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 8 \
               --num-layers 80 \
               --hidden-size 8192 \
               --ffn-hidden-size 22016 \
               --num-attention-heads 64 \
               --micro-batch-size 2 \
               --global-batch-size 512 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --exit-interval 2 \
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
               --apply-rotary-positional-embedding-kernel \
               --rmsnorm-epsilon 1e-6 \
               --no-position-embedding \
               --no-masked-softmax-fusion \
               --no-gradient-accumulation-fusion \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --untie-embeddings-and-output-weights \
               --initial-loss-scale 16 \
               --no-contiguous-buffers-in-local-ddp \
               --sequence-parallel \
               --recompute-granularity full \
               --recompute-method uniform \
               --recompute-num-layers 5 \
               --timing-log-level 2\
               "

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 100 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"

TENSORBOARD_ARGS="--tensorboard-dir $TENSORBOARD_PATH \
                  --tensorboard-queue-size 5 \
                  --log-timers-to-tensorboard \
                  --log-batch-size-to-tensorboard \
                  --log-validation-ppl-to-tensorboard"

# debug
# export BKCL_DEBUG=1
# export XLOG_LEVEL="dist=info"

PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA_65B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
