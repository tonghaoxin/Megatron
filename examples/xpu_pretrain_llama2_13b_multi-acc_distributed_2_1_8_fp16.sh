#! /bin/bash

set -e
set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )


#################################
# 清理环境
#################################
pkill -9 python || true

#################################
# 数据集部分
#################################
DATA_PATH=/mnt/data1/megatron_datasets/llama/data/llama_00_text_document/llama_00_text_document
#CHECKPOINT_LOAD_PATH=/mnt/data1/megatron_datasets/llama/checkpoints/megatron_llama_13b_checkpoint_tp8_pp1_fp32
TOKENIZER_MODEL_PATH=/mnt/data1/megatron_datasets/llama/tokenizer_model/tokenizer.model


# 多机bkcl 需要0506之后版本, 
export LD_LIBRARY_PATH=/mnt/data1/70b_bkcl_so/0425:/mnt/data1/xre-Linux-x86_64-0.0.0.1-2024-04-16-23-03-29-daily/so:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`


#################################
# 多机MEGATRON设置部分
#################################
# 内网地址方便互联：
# p800-1 i-4hhrz3KF: 10.7.64.87      172.16.0.2
# p800-2 i-VtFmCOWF：100.72.0.214    172.16.0.8
# p800-3 i-ecQvCmDM: 10.7.65.221     172.16.0.9
# p800-4 i-XlM4xYtb: 100.74.0.204    172.16.0.3

# 当前多机配置，需要设定一个MASTER_ADDR, 用来两台机器通过MASTER来沟通，下面以214机器的内网ip作为MASTER_ADDR
MASTER_ADDR=${MASTER_ADDR:-"172.16.0.8"}
MASTER_PORT=${MASTER_PORT:-9909}

# 当前多机配置，这里设置两机， rank 根据情况自己设置调整
NNODES=${1:-2}
NODE_RANK=${2:-0} # RANK 根据当前机器设置调整

ulimit -c 0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3


#################################
# 多机KL3环境变量
#################################

export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_BUFFER_GM=1
export BKCL_RING_FORCE_BULK=0 # 多机关闭

# rdma 需要设置
export BKCL_FORCE_L3_RDMA=0
export BKCL_ENABLE_XDR=1  
export BKCL_RDMA_FORCE_TREE=1 
export BKCL_TREE_THRESHOLD=1

# 需要视机器情况来做修改(命令 ip a查看)
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4 
# export BKCL_RDMA_NICS=xgbe1,xgbe1,xgbe2,xgbe2,xgbe3,xgbe3,xgbe4,xgbe4

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

# 检查 /dev/xdrdrv 是否存在
if [ ! -e "/dev/xdrdrv" ]; then
    echo "Error: /dev/xdrdrv not found! PLEASE CHECK YOUR DOCKER CONTAINER"
    exit 1
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA2_13B_ARGS="--tensor-model-parallel-size 2 \
                --pipeline-model-parallel-size 1 \
                --num-layers 40 \
                --hidden-size 5120 \
                --ffn-hidden-size 13824 \
                --num-attention-heads 40 \
                --micro-batch-size 2 \
                --global-batch-size 1024 \
                --exit-interval 300 \
                --seq-length 4096 \
                --max-position-embeddings 4096 \
                --lr 0.0001 \
                --min-lr 1.0e-5 \
                --clip-grad 1.0 \
                --optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.95 \
                --weight-decay 0.01 \
                --adam-eps 1e-05 \
                --train-iters 500000 \
                --lr-decay-iters 320000 \
                --lr-decay-style cosine \
                --lr-warmup-fraction .01 \
                --no-async-tensor-model-parallel-allreduce \
                --tokenizer-type LLaMASentencePieceTokenizer \
                --tokenizer-model $TOKENIZER_MODEL_PATH \
                --activation-func swiglu \
                --use-rotary-position-embeddings \
                --rmsnorm-epsilon 1e-5 \
                --no-position-embedding \
                --no-masked-softmax-fusion \
                --disable-bias-linear \
                --attention-dropout 0 \
                --hidden-dropout 0 \
                --embedding-dropout 0 \
                --untie-embeddings-and-output-weights \
                --initial-loss-scale 16 \
                --use-distributed-optimizer \
                --memory-saving \
                --fused-rmsnorm \
                --apply-rotary-positional-embedding-kernel \
                --use-cpu-initialization \
                --sequence-parallel \
                --use-flash-attn \
                --fp16"


OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"

#################################
# driver方式部分
#################################
#pmode 模式需要开启下面环境变量
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为

#imode 模式需要同时开启下列3个环境变量
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为
export XPU_FORCE_USERMODE_LAUNCH=1 #强制使用纯用户态高性能launch/it模式，可以获得理论最小的launch开销，需要kunlun.ko insmod时使用
# export XPU_DUMMY_EVENT=1 #强制event相关API返回成功，当前用于event还未ready的纯用户态高性能launch/wait模式
# 如果要开启多流 则需要 关掉 XPU_DUMMY_EVENT
export DIST_MULTI_STREAM=true
#################################
# 算子部分
#################################
#export FAST_SWIGLU_ENABLE=1 # 查表SWIGLU算子实现
export XMLIR_FA_GEMM_TYPE=float16 # FA GEMM type默认为tf32，较慢，需要指定为fp16
#export XPUAPI_DEFAULT_SIZE=4000000000 # 减少算子重新预分配内存导致的 XPU_WAIT开销, 副作用是额外多消耗显存
#export XBLAS_FC_HBM_VERSION=40 # 需要指定HBM版 from jiahui
#################################
# 算子检查部分
#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # 禁止XDNN出错后fallback CPU
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #禁止任何fallback
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

#################################
# hbm部分
#################################
# xccl 多机的时候会额外分配 L3 memory， 需要脚本中禁掉其他L3 memory的使用
# export XPU_FORCE_CODE_PARAM_LOCATE_IN_L3=1 #高频hbm必开, stable/low选开

#################################
# BKCL C2C部分
#################################
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1

# ccix_inner_8chips
cat > ccix_inter.txt <<EOF
[chip 0, port 0] <===> [chip 6, port 3]
[chip 4, port 1] <===> [chip 5, port 2]
[chip 2, port 3] <===> [chip 4, port 0]
[chip 2, port 1] <===> [chip 3, port 2]
[chip 1, port 1] <===> [chip 3, port 1]
[chip 0, port 1] <===> [chip 1, port 2]
[chip 0, port 2] <===> [chip 2, port 2]
[chip 0, port 3] <===> [chip 3, port 3]
[chip 1, port 0] <===> [chip 2, port 0]
[chip 1, port 3] <===> [chip 7, port 0]
[chip 5, port 1] <===> [chip 7, port 1]
[chip 3, port 0] <===> [chip 5, port 3]
[chip 4, port 3] <===> [chip 7, port 3]
[chip 4, port 2] <===> [chip 6, port 2]
[chip 6, port 1] <===> [chip 7, port 2]
[chip 5, port 0] <===> [chip 6, port 0]
EOF

# cat > ccix_inter.txt <<EOF
# [chip 3, port 0] <===> [chip 7, port 3]
# [chip 3, port 1] <===> [chip 2, port 2]
# [chip 3, port 2] <===> [chip 1, port 2]
# [chip 3, port 3] <===> [chip 0, port 3]
# [chip 2, port 0] <===> [chip 1, port 0]
# [chip 2, port 1] <===> [chip 0, port 1]
# [chip 2, port 3] <===> [chip 6, port 0]
# [chip 1, port 1] <===> [chip 0, port 2]
# [chip 1, port 3] <===> [chip 5, port 0]
# [chip 0, port 0] <===> [chip 4, port 3]
# [chip 5, port 1] <===> [chip 4, port 2]
# [chip 5, port 2] <===> [chip 7, port 2]
# [chip 5, port 3] <===> [chip 6, port 3]
# [chip 4, port 0] <===> [chip 7, port 0]
# [chip 4, port 1] <===> [chip 6, port 1]
# [chip 7, port 1] <===> [chip 6, port 2]
# EOF


export XPU_ZEBU_MODE=1 # 影响ccix_inter.txt文件读取
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_RING_BUFFER_GM=1
# export BKCL_FORCE_SYNC=1
export BKCL_KL3_TURBO_MODE=1 
export BKCL_RING_BUFFER_SIZE=2097152
#export XPUSIM_TOPOLOGY_FILE="ccix_inter.txt"
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000
unset BKCL_KL3_SYSCON_FLAG


#################################
# DEBUG 用途部分
#################################
# export XLOG_LEVEL="capture=info,dist=info"
# export XPURT_DISPATCH_MODE=PROFILING
# export XPUAPI_DEBUG=0x1001

#export PYTORCH_NO_CUDA_MEMORY_CACHING=1

#export BKCL_PERF_DEBUG=1
#export BKCL_DEBUG=1
#export BKCL_DUMP=1
#export BKCL_DEBUG_SKIP_RUN=1

#export CUDA_DEVICE_MAX_CONNECTIONS=1


#################################
# torchrun 测试
#################################
# torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 9999 torchrun_8x1_test.py



#################################
# XFLAGS插件设置
#################################
XFLAGS --list
XFLAGS --enable megatron_aiak
XFLAGS --disable megatron_23_05

#################################
# 13B 模型训练
#################################
PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH torchrun $DISTRIBUTED_ARGS $ROOT_DIR/../pretrain_llama.py $LLAMA2_13B_ARGS $OUTPUT_ARGS $OTHER_ARGS