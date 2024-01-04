#! /bin/bash

set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

#data数据路径记得修改
DATA_PATH=$ROOT_DIR/../data/alpaca_data_en_52k.json
DATA_DIR=$ROOT_DIR/../data/
TOKENIZER_PATH=$ROOT_DIR/../tokenizer/tokenizer.model
CHECKPOINT_PATH=${ROOT_DIR}/../checkpoints_8_4_1/

export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$XPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_GID_INDEX=3

export BKCL_CCIX_BUFFER_GM=1
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1

export BKCL_SOCKET_IFNAME=$SOCKET_IFNAME
export BKCL_USE_RDMA=1
#export BKCL_FORCE_L3_RDMA=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_ENABLE_XDR=0
export BKCL_RING_BUFFER_SIZE=1024000
export BKCL_RDMA_NICS=$RDMA_NICS
export BKCL_FORCE_ALLREDUCE_IN_MULTINODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA2_13B_ARGS="--seed 42 \
                 --preprocessing-num-workers 1 \
                 --tensor-model-parallel-size 8 \
                 --pipeline-model-parallel-size 4 \
                 --num-layers 40 \
                 --hidden-size 5120 \
                 --ffn-hidden-size 13824 \
                 --num-attention-heads 40 \
                 --micro-batch-size 1 \
                 --global-batch-size 128 \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --lr 5e-6 \
                 --clip-grad 1.0 \
                 --weight-decay 0. \
                 --train-iters 1216 \
                 --lr-decay-style cosine \
                 --lr-warmup-fraction .03 \
                 --no-async-tensor-model-parallel-allreduce \
                 --tokenizer-type LLaMASFTTokenizerHF \
                 --tokenizer-model $TOKENIZER_PATH \
                 --activation-func swiglu \
                 --use-rotary-position-embeddings \
                 --rmsnorm-epsilon 1e-5 \
                 --no-position-embedding \
                 --disable-bias-linear \
                 --untie-embeddings-and-output-weights \
                 --attention-dropout 0 \
                 --hidden-dropout 0 \
                 --embedding-dropout 0 \
                 --multi-query-attention \
                 --multi-query-group-num 40 \
                 --sft-collator \
                 --dataloader-type cyclic \
                 --sequence-parallel \
                 --no-gradient-accumulation-fusion \
                 --recompute-activations"

OUTPUT_ARGS="--log-interval 1 \
            --save-interval 10000 \
            --eval-interval 1000 \
            --eval-iters 10"

OTHER_ARGS="--data-path $DATA_PATH \
            --dataset-dir $DATA_DIR \
            --load $CHECKPOINT_PATH \
           --data-impl mmap \
           --split 949,50,1 \
           --distributed-backend nccl"

ulimit -c 0
export XMLIR_F_XPU_ENABLED_BOOL=true
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=1800
# export BKCL_SOCKET_IFNAME=ens20f0

# debug
#export BKCL_DEBUG=1
#export XLOG_LEVEL="dist=info"

# profile xdnn_pytorch
#export XPURT_DISPATCH_MODE=PROFILING
#export XPUAPI_DEBUG=0x1001

# profile api scale
#export XPUAPI_DEBUG=0x1

# use pcie
#export BKCL_PCIE_RING=1

# use ccix
# export BKCL_CCIX_RING=1
# export BKCL_TREE_THRESHOLD=1

# # bckl not use L3
# export BKCL_CCIX_BUFFER_GM=1
#export BKCL_RING_BUFFER_SIZE=524288

# xmlir set L3 size, default 13M
#export XMLIR_D_XPU_L3_SIZE=1048576
export XDNN_FC_GEMM_DTYPE=float32

export XACC=1
PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../sft_llama.py \
       $LLAMA2_13B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
