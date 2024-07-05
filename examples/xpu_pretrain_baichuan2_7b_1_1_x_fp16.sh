#! /bin/bash

set -x

ulimit -c
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

#data数据路径记得修改
DATA_PATH=${BAICHUAN_7B_DATA:-"/home/data"}/pile_baichuan2_text_document
echo "DATA_PATH: $DATA_PATH"

# TOKENIZER 目录
TOKENIZER_PATH=${BAICHUAN_7B_TOKEN:-"/home/tokenizer"}/tokenizer.model
echo "TOKENIZER_PATH: $TOKENIZER_PATH"

ITERATION_NUM=${ITERATION:-5}

GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9988}
NNODES=${1:-1}
NODE_RANK=${2:-0}

LAYER_NUM=32
if [ $GPUS_PER_NODE -eq 1 ]; then
       LAYER_NUM=16
fi
echo "LAYER_NUM: $LAYER_NUM"

############################################ Kernel Launch Mode Configuration Begin        ############################################
#imode 模式需要同时开启下列3个环境变量
export XPU_FORCE_USERMODE_LAUNCH=1 #强制使用纯用户态高性能launch/it模式，可以获得理论最小的launch开销，需要kunlun.ko insmod时使用
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为
#unset XPU_FORCE_USERMODE_LAUNCH
unset XPU_DUMMY_EVENT
############################################ Kernel Launch Mode Configuration End          ############################################

############################################ Communication Library Configuration Begin     ############################################
# export NCCL_DEBUG=INFO # for debug

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0

export BKCL_TREE_THRESHOLD=0

export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0

unset BKCL_KL3_SYSCON_FLAG
############################################ Communication Library Configuration End       ############################################

############################################ Op Performance Configuration Begin            ############################################
# Op performance env
export XPUAPI_DEFAULT_SIZE=4000000000
export XBLAS_FC_HBM_VERSION=40 # 需要指定HBM版
export XMLIR_FA_GEMM_TYPE=float16 #FA GEMM type默认为tf32，较慢，需要指定为fp16
############################################ Op Performance Configuration End              ############################################

############################################ XMLIR Configuration Begin                     ############################################
# Prohibit op fallback to cpu
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # 禁止XDNN出错后fallback CPU
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #禁止任何fallback
export DIST_MULTI_STREAM=true
############################################ XMLIR Configuration End                       ############################################


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

BAICHUAN2_7B_ARGS="--tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 1 \
               --num-layers $LAYER_NUM \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 2 \
               --global-batch-size 1024 \
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
               --tokenizer-type BaichuanTokenizer \
               --tokenizer-model $TOKENIZER_PATH \
               --activation-func swiglu \
               --use-rotary-position-embeddings \
               --rmsnorm-epsilon 1e-5 \
               --no-position-embedding \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --untie-embeddings-and-output-weights \
               --use-distributed-optimizer\
               --initial-loss-scale 524288 \
               --fp16 \
               --use-flash-attn \
               --use-cpu-initialization \
               --memory-saving \
               --no-gradient-accumulation-fusion \
               --no-bias-gelu-fusion \
               --fused-rmsnorm \
               --apply-rotary-positional-embedding-kernel \
               --exit-interval $ITERATION_NUM \
               --no-masked-softmax-fusion"

OUTPUT_ARGS="--log-interval 1 \
             --eval-interval 1000 \
             --eval-iters 10"
             #--save-interval 100 \
             #--save $CHECKPOINT_SAVE_PATH \

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"

XFLAGS --enable megatron_aiak

PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_baichuan.py \
       $BAICHUAN2_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
