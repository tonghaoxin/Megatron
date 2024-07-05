#!/bin/bash

set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

#################################
# 清理环境
#################################
pkill -9 python || true

############################################ Parameters Configuration Begin                ############################################
# 1. data数据路径记得修改
DATA_PATH=${QWEN_14B_DATA:-"/home/data"}/qwen_00_text_document_content_sentence
echo "DATA_PATH: $DATA_PATH"

# 2. TOKENIZER 目录
TOKENIZER_PATH=${QWEN_14B_TOKEN:-"/home/tokenizer"}
echo "TOKENIZER_PATH: $TOKENIZER_PATH"

# 3. BKCL_SO
BKCL_SO=${BKCL:-"/home/xccl/so"}
echo "BKCL_SO: $BKCL_SO"

# 4. RUNTIME_SO
RUNTIME_SO=${RUNTIME:-"/home/runtime/so"}
echo "RUNTIME_SO: $RUNTIME_SO"

export LD_LIBRARY_PATH=$BKCL_SO:$RUNTIME_SO:$LD_LIBRARY_PATH

ITERATION_NUM=${ITERATION:-5}

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${NNODES:-2}
# NODE_RANK按需修改
NODE_RANK=${NODE_RANK:-0}

ulimit -c 0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

QWEN_14B_ARGS="--tensor-model-parallel-size 1
               --pipeline-model-parallel-size 2 \
               --num-layers 40 \
               --hidden-size 5120 \
               --ffn-hidden-size 13696 \
               --num-attention-heads 40 \
               --init-method-std 0.01 \
               --micro-batch-size 2 \
               --global-batch-size 128 \
               --seq-length 4096 \
               --max-position-embeddings 4096 \
               --tokenizer-type QwenTokenizer \
               --tokenizer-model $TOKENIZER_PATH/qwen.tiktoken \
               --lr 1.5e-4 \
               --min-lr 1.5e-5 \
               --lr-decay-style cosine \
               --weight-decay 1e-2 \
               --clip-grad 1.0 \
               --initial-loss-scale 64 \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --train-iters 500000 \
	           --lr-decay-iters 320000 \
               --lr-warmup-fraction 0.01 \
               --sequence-parallel  \
               --rmsnorm-epsilon 1e-6 \
               --activation-func swiglu \
               --use-distributed-optimizer \
               --use-rotary-position-embeddings \
               --untie-embeddings-and-output-weights \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --attention-dropout 0 \
               --memory-saving  \
               --disable-bias-linear \
               --no-bias-gelu-fusion \
               --no-position-embedding \
               --no-masked-softmax-fusion \
               --no-masked-softmax-fusion \
               --no-query-key-layer-scaling \
               --attention-softmax-in-fp32 \
               --memory-saving \
               --fused-rmsnorm \
	           --use-flash-attn \
               --optimizer adam \
               --use-cpu-initialization \
               --exit-interval $ITERATION_NUM \
               --fp16"


OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 98,2,0 \
            --distributed-backend nccl"


#################################
# driver方式部分
#################################
export DIST_MULTI_STREAM=true
#imode 模式需要同时开启下列3个环境变量
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为
export XPU_FORCE_USERMODE_LAUNCH=1 #强制使用纯用户态高性能launch/it模式，可以获得理论最小的launch开销，需要kunlun.ko insmod时使用
#export XPU_DUMMY_EVENT=1 #强制event相关API返回成功，当前用于event还未ready的纯用户态高性能launch/wait模式


#################################
# 算子部分
#################################
# type默认为tf32，较慢，需要指定为fp16
export XMLIR_FA_GEMM_TYPE=float16
export FAST_SWIGLU_ENABLE=1
export XPUAPI_DEFAULT_SIZE=3000000000 # 减少算子重新预分配内存导致的 XPU_WAIT开销, 副作用是额外多消耗显存
export XBLAS_FC_HBM_VERSION=40

#################################
# 算子检查部分
#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # 禁止XDNN出错后fallback CPU
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #禁止任何fallback
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

#################################
# hbm部分
#################################
export XPU_FORCE_CODE_PARAM_LOCATE_IN_L3=1 #高频hbm必开, stable/low选开


#################################
# BKCL C2C部分
#################################
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1


export XPU_ZEBU_MODE=1 # 影响ccix_inter.txt文件读取
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_RING_BUFFER_GM=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152
# export XPUSIM_TOPOLOGY_FILE="ccix_inter.txt"
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000
unset BKCL_KL3_SYSCON_FLAG

echo "whosyourdaddy" >> /proc/kunlun/dev0/info
echo "whosyourdaddy" >> /proc/kunlun/dev1/info
echo "whosyourdaddy" >> /proc/kunlun/dev2/info
echo "whosyourdaddy" >> /proc/kunlun/dev3/info
echo "whosyourdaddy" >> /proc/kunlun/dev4/info
echo "whosyourdaddy" >> /proc/kunlun/dev5/info
echo "whosyourdaddy" >> /proc/kunlun/dev6/info
echo "whosyourdaddy" >> /proc/kunlun/dev7/info
echo "3600000" > /proc/kunlun/dev0/task_timeout_detect_threshold_in_ms
echo "3600000" > /proc/kunlun/dev1/task_timeout_detect_threshold_in_ms
echo "3600000" > /proc/kunlun/dev2/task_timeout_detect_threshold_in_ms
echo "3600000" > /proc/kunlun/dev3/task_timeout_detect_threshold_in_ms
echo "3600000" > /proc/kunlun/dev4/task_timeout_detect_threshold_in_ms
echo "3600000" > /proc/kunlun/dev5/task_timeout_detect_threshold_in_ms
echo "3600000" > /proc/kunlun/dev6/task_timeout_detect_threshold_in_ms
echo "3600000" > /proc/kunlun/dev7/task_timeout_detect_threshold_in_ms

#################################
# 14B 模型训练
#################################
XFLAGS --list
XFLAGS --enable megatron_aiak

PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_qwen.py \
       $QWEN_14B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
