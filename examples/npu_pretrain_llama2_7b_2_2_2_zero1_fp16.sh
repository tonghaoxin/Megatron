#! /bin/bash

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
TOKENIZER_PATH=$ROOT_DIR/../tokenizer/tokenizer.model
CHECKPOINT_PATH=${ROOT_DIR}/../checkpoints/
# VOCAB_FILE=${ROOT_DIR}/../data/gpt2-vocab.json
# MERGE_FILE=${ROOT_DIR}/../data/gpt2-merges.txt

export NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPUS_PER_NODE=`echo "$NPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`
# Change for multinode config
MASTER_PORT=${MASTER_PORT:-8658}
MASTER_ADDR=${1:-"127.0.0.1"}
NNODES=${2:-1}
NODE_RANK=${3:-0}

# export OMP_NUM_THREADS=96
# export ASCEND_GLOBAL_LOG_LEVEL=3
# export ASCEND_GLOBAL_EVENT_ENABLE=1
# export INF_NAN_MODE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=1200
# export HCCL_INTRA_PCIE_ENABLE=0
# export HCCL_INTRA_ROCE_ENABLE=1
# export HCCL_RDMA_TC=128
# export HCCL_RDMA_SL=4
export COMBINED_ENABLE=1
# export HCCL_SOCKET_IFNAME=xgbe0
export MULTI_STREAM_MEMORY_REUSE=1
# export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA2_7B_ARGS="--tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 2 \
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
                --no-position-embedding \
                --disable-bias-linear \
                --attention-dropout 0 \
                --hidden-dropout 0 \
                --embedding-dropout 0 \
                --sequence-parallel
                --initial-loss-scale 16 \
                --use-distributed-optimizer \
                --untie-embeddings-and-output-weights \
                --apply-rotary-positional-embedding-kernel \
                --use-flash-attn \
                --fused-rmsnorm \
                --fp16"

HETERO_ARGS="--hetero-mode pp \
             --hetero-cross-platform \
             --hetero-pipeline-stages 1 16 1 16 \
             --hetero-device-types gpu npu \
             --hetero-current-device-type npu"
             # --hetero-micro-batch-sizes 2 1 2 1 \

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--save $CHECKPOINT_PATH \
            --load $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 100,0,0 \
            --distributed-backend nccl"

PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/npu_ops':$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA2_7B_ARGS \
       $HETERO_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
