#!/bin/bash

'''
# 测试数据集 BOS 地址
bos://cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/llama_test/
# llama tokenizer model BOS 地址
bos://cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/llama_tokenizer/
# 7B 模型 BOS 地址
bos://cce-ai-datasets/cce-ai-datasets.bj.bcebos.com/megatron_llama/megatron_llama_7b_checkpoint_tp8_pp1/
'''

set -x

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/opt/rh/devtoolset-7/root/usr/bin
export LD_LIBRARY_PATH=/usr/local/python3.8.18/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.8.18/bin:$PATH
export PATH=/usr/local/python3.8.18/bin/pip:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

DATA_PATH=${ROOT_DIR}/../data/llama_00_text_document
TOKENIZER_PATH=${ROOT_DIR}/../tokenizer/tokenizer.model
CHECKPOINT_LOAD_PATH=${ROOT_DIR}/../checkpoints/megatron_llama_7b_checkpoint_tp8_pp1_fp16
CHECKPOINT_SAVE_PATH=${ROOT_DIR}/../checkpoints/megatron_llama_7b_checkpoint_tp8_pp1_fp32_save
TENSORBOARD_PATH='./llama7b_tp8_pp1_fp16_tensorboard'

#export NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPUS_PER_NODE=`echo "$NPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}

export INF_NAN_MODE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_RDMA_TC=128
export HCCL_RDMA_SL=4
export COMBINED_ENABLE=1
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_GLOBAL_EVENT_ENABLE=0
export OMP_NUM_THREADS=96
export HCCL_SOCKET_IFNAME=xgbe0

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA_7B_ARGS="--tensor-model-parallel-size 8 \
               --pipeline-model-parallel-size 1 \
               --num-layers 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 1 \
               --global-batch-size 128 \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --lr 1.0e-5 \
               --min-lr 1.0e-5 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --adam-eps 1e-08 \
               --weight-decay 0.1 \
               --train-iters 500000 \
               --lr-decay-iters 320000 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .01 \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-type LLaMASentencePieceTokenizer \
               --tokenizer-model $TOKENIZER_PATH \
               --init-method-std 0.01 \
               --activation-func swiglu \
               --use-rotary-position-embeddings \
               --rmsnorm-epsilon 1e-6 \
               --no-position-embedding \
               --no-masked-softmax-fusion \
               --no-gradient-accumulation-fusion \
               --untie-embeddings-and-output-weights \
               --make-vocab-size-divisible-by 128 \
               --no-bias-gelu-fusion \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --initial-loss-scale 16 \
               --use-distributed-optimizer \
               --reset-attention-mask \
               --sequence-parallel \
               --num-workers 64 \
               --bf16"
               #--apply-rotary-positional-embedding-kernel
               #--use-fused-rotary-pos-emb \
               #--use-flash-attn \
               #--fused-rmsnorm \

               #--tensorboard-dir $TENSORBOARD_PATH \
               #--tensorboard-queue-size 5 \
               #--log-timers-to-tensorboard \
               #--log-batch-size-to-tensorboard \
               #--log-validation-ppl-to-tensorboard"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 50000 \
             --eval-interval 50000 \
             --eval-iters 0"

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --DDP-impl local \
            --distributed-backend nccl"
            #--load $CHECKPOINT_LOAD_PATH \
            #--save $CHECKPOINT_SAVE_PATH \

PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
