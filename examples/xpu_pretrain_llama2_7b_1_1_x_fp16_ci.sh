#!/bin/bash

set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

############################################ Parameters Configuration Begin                ############################################
# 1. data数据路径记得修改
DATA_PATH=${DATA:-"/home/data"}/pile-llama_text_document
echo "DATA_PATH: $DATA_PATH"

# 2. TOKENIZER 目录
TOKENIZER_PATH=${TOKEN:-"/home/tokenizer"}
echo "TOKENIZER_PATH: $TOKENIZER_PATH"

# 3. BKCL_SO
BKCL_SO=${BKCL:-"/home/xccl/so"}
echo "BKCL_SO: $BKCL_SO"

# 4. RUNTIME_SO
RUNTIME_SO=${RUNTIME:-"/home/runtime/so"}
echo "RUNTIME_SO: $RUNTIME_SO"

export LD_LIBRARY_PATH=$BKCL_SO:$RUNTIME_SO:$LD_LIBRARY_PATH

ITERATION_NUM=${ITERATION:-5}

GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

LAYER_NUM=32
if [ $GPUS_PER_NODE -eq 1 ]; then
       LAYER_NUM=16
fi
echo "LAYER_NUM: $LAYER_NUM"

############################################ Parameters Configuration End                  ############################################


############################################ Kernel Launch Mode Configuration Begin        ############################################
#imode 模式需要同时开启下列3个环境变量
export XPU_FORCE_USERMODE_LAUNCH=1 #强制使用纯用户态高性能launch/it模式，可以获得理论最小的launch开销，需要kunlun.ko insmod时使用
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为
unset XPU_DUMMY_EVENT
############################################ Kernel Launch Mode Configuration End          ############################################


############################################ Communication Library Configuration Begin     ############################################
# export NCCL_DEBUG=INFO # for debug
# export NCCL_DEBUG_SUBSYS=ALL

export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=3600

export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1

export XPU_ZEBU_MODE=1 # 影响ccix_inter.txt文件读取
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0

export BKCL_RING_BUFFER_GM=1
unset BKCL_KL3_SYSCON_FLAG
############################################ Communication Library Configuration End       ############################################


############################################ Op Performance Configuration Begin            ############################################
# Op performance env
export XPUAPI_DEFAULT_SIZE=4000000000
export XBLAS_FC_HBM_VERSION=40 # 需要指定HBM版
export XPU_FORCE_CODE_PARAM_LOCATE_IN_L3=1 #高频hbm必开, stable/low选开
export XMLIR_FA_GEMM_TYPE=float16 #FA GEMM type默认为tf32，较慢，需要指定为fp16
export FAST_SWIGLU_ENABLE=1 # 查表SWIGLU算子实现
############################################ Op Performance Configuration End              ############################################


############################################ XMLIR Configuration Begin                     ############################################
# Prohibit op fallback to cpu 
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # 禁止XDNN出错后fallback CPU
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #禁止任何fallback
############################################ XMLIR Configuration End                       ############################################


############################################ Other Begin                                   ############################################
# profile xdnn_pytorch
# export XPURT_DISPATCH_MODE=PROFILING
#For Multi-Stream
#export DIST_MULTI_STREAM=true
############################################ Other End                                     ############################################


ulimit -c 0

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA_7B_ARGS="--tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 1 \
               --num-layers $LAYER_NUM \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --init-method-std 0.01 \
               --micro-batch-size 1 \
               --global-batch-size 128 \
               --seq-length 4096 \
               --max-position-embeddings 4096 \
               --pad-vocab-size-to 32000 \
               --make-vocab-size-divisible-by 128 \
               --tokenizer-type LLaMASentencePieceTokenizer \
               --tokenizer-model $TOKENIZER_PATH/tokenizer.model \
               --lr 1.5e-4 \
               --min-lr 1.5e-5 \
               --lr-decay-style cosine \
               --weight-decay 1e-2 \
               --clip-grad 1.0 \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --train-iters 11920 \
               --lr-decay-iters 10251 \
               --lr-warmup-iters 2000 \
               --rmsnorm-epsilon 1e-6 \
               --activation-func swiglu \
               --use-rotary-position-embeddings \
               --untie-embeddings-and-output-weights \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --disable-bias-linear \
               --no-bias-gelu-fusion \
               --no-position-embedding \
               --no-masked-softmax-fusion \
               --attention-softmax-in-fp32 \
               --no-query-key-layer-scaling \
               --initial-loss-scale 64 \
               --use-flash-attn \
               --fused-rmsnorm \
               --memory-saving \
               --use-cpu-initialization \
               --use-distributed-optimizer \
               --exit-interval $ITERATION_NUM \
               --fp16"

OUTPUT_ARGS="--log-interval 1 \
             --eval-interval 1000 \
             --eval-iters 10"
             #--save-interval 100 \
             #--save $CHECKPOINT_SAVE_PATH \

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"


PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS

