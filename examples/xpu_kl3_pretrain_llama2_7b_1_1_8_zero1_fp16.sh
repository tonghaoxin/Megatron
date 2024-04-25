#! /bin/bash

set -e
set -x

#export BKCL_PERF_DEBUG=1
#export FAST_SWIGLU_ENABLE=1
#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
#export XPURT_DISPATCH_MODE=PROFILING
#export XPUAPI_DEBUG=0x1
#export _XMLIR_D_XDNN_CPU_FALLBACK_OP_STR=uniform_
#export BKCL_PERF_DEBUG=1
#export BKCL_DEBUG=1
#export BKCL_DUMP=1
#export BKCL_DEBUG_SKIP_RUN=1

#export CUDA_DEVICE_MAX_CONNECTIONS=1


#export XMLIR_FC_AUTOTUNE_BOOL=false
#export XMLIR_FC_AUTOTUNE_FILE_NAME=/home/fengyaping/Megatron-latest/examples/llama7b_tune_result.txt

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

#data数据路径记得修改
DATA_PATH=$ROOT_DIR/../data/alpaca_text_document
TOKENIZER_PATH=$ROOT_DIR/../tokenizer/tokenizer.model
CHECKPOINT_LOAD_PATH=$ROOT_DIR/../checkpoints/megatron_llama2_7b_checkpoint_tp1_pp1_dp8_zero1_fp16
CHECKPOINT_SAVE_PATH=$ROOT_DIR/../checkpoints/megatron_llama2_7b_checkpoint_tp1_pp1_dp8_zero1_fp16_saved

export LD_LIBRARY_PATH=$ROOT_DIR/../xccl-so:$ROOT_DIR/../xre-so:$LD_LIBRARY_PATH

# bkcl new, runtime old
#export LD_LIBRARY_PATH=/workspace/13b/bkcl_so_20240306_ipipe/output/xccl_socket_ubuntu_x86_64/so:/workspace/13b/runtime-close/runtime_kl3_515.58_20240222165034/so_relwithdebinfo:$LD_LIBRARY_PATH

# bckl new, runtime 0306 new
#export LD_LIBRARY_PATH=/workspace/13b/bkcl_so_20240306_ipipe/output/xccl_socket_ubuntu_x86_64/so:/workspace/13b/515_58_runtime_so_20240306/runtime-close/runtime_kl3_515.58_20240305113144/so_relwithdebinfo:$LD_LIBRARY_PATH

#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
#export LD_LIBRARY_PATH=/workspace/13b/bkcl_so_20240306:/workspace/13b/515_58_runtime_so_20240306_update_ver1/api:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/workspace/13b/bkcl_so_20240306:/workspace/13b/runtime-close/runtime_kl3_515.58_20240222165034/so_relwithdebinfo:$LD_LIBRARY_PAT
#export LD_LIBRARY_PATH=/workspace/13b/bkcl_so_20240306_ipipe/output/xccl_socket_ubuntu_x86_64/so:/workspace/13b/runtime-close/runtime_kl3_515.58_20240222165034/so_relwithdebinfo:$LD_LIBRARY_PAT
#export LD_LIBRARY_PATH=/workspace/13b/bkcl_so_20240306_ipipe/output/xccl_socket_ubuntu_x86_64/so:/workspace/13b/runtime-close/runtime_kl3_515.58_20240222165034/so_relwithdebinfo:$LD_LIBRARY_PAT

#export BKCL_TREE_THRESHOLD=0
#export XPU_ZEBU_MODE=1
#export BKCL_XLINK_C2C=1
#export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
#export BKCL_CLUSTERS_PER_RING=2
#export BKCL_RING_BUFFER_SIZE=2048000
#export XPUSIM_TOPOLOGY_FILE="ccix_inter.txt"

export XPUAPI_DEFAULT_SIZE=4000000000
#pmode
export CUDART_DUMMY_REGISTER=1 #强制__cudaRegister***返回成功，用绕过pytorch混用NV官方cublas/cudnn造成的奇怪行为

#imode 模式需要同时开启下列3个环境变量
export XPU_FORCE_USERMODE_LAUNCH=1 #强制使用纯用户态高性能launch/it模式，可以获得理论最小的launch开销，需要kunlun.ko insmod时使用
export XPU_DUMMY_EVENT=1 #强制event相关API返回成功，当前用于event还未ready的纯用户态高性能launch/wait模式

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${1:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${2:-1}
NODE_RANK=${3:-0}

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export XMLIR_F_XPU_ENABLED_BOOL=true
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=3600
#export BKCL_SOCKET_IFNAME=bond0

# debug
# export BKCL_DEBUG=1
#export XLOG_LEVEL="dist=info"


# profile xdnn_pytorch
# export XPURT_DISPATCH_MODE=PROFILING
export XPU_FORCE_CODE_PARAM_LOCATE_IN_L3=1 #高频hbm必开, stable/low选开
export XMLIR_FA_GEMM_TYPE=float16 #FA GEMM type默认为tf32，较慢，需要指定为fp16
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # 禁止XDNN出错后fallback CPU
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #禁止任何fallback


# use ccix
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1

# bckl not use L3
export BKCL_CCIX_BUFFER_GM=1
#export BKCL_RING_BUFFER_SIZE=524288

# xmlir set L3 size, default 13M``
#export XMLIR_D_XPU_L3_SIZE=1048576
#export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

# 需要根据真机的实际拓扑调整ccix_inter.txt 内容 .否则会出错
#cat > ccix_inter.txt <<EOF
#[chip 0, port 1] <===> [chip 1, port 2]
#EOF


#88 ccix_inner_4chips
cat > ccix_inter.txt <<EOF
[chip 3, port 0] <===> [chip 7, port 3]
[chip 3, port 1] <===> [chip 2, port 2]
[chip 3, port 2] <===> [chip 1, port 2]
[chip 3, port 3] <===> [chip 0, port 3]
[chip 2, port 0] <===> [chip 1, port 0]
[chip 2, port 1] <===> [chip 0, port 1]
[chip 2, port 3] <===> [chip 6, port 0]
[chip 1, port 1] <===> [chip 0, port 2]
[chip 1, port 3] <===> [chip 5, port 0]
[chip 0, port 0] <===> [chip 4, port 3]
[chip 5, port 1] <===> [chip 4, port 2]
[chip 5, port 2] <===> [chip 7, port 2]
[chip 5, port 3] <===> [chip 6, port 3]
[chip 4, port 0] <===> [chip 7, port 0]
[chip 4, port 1] <===> [chip 6, port 1]
[chip 7, port 1] <===> [chip 6, port 2]
EOF



#cat > ccix_inter.txt <<EOF
#[chip 2, port 1] <===> [chip 3, port 2]
#EOF

#export XPUSIM_RUN_KERNEL_NAME=range/cast
export XPU_ZEBU_MODE=1 # 影响ccix_inter.txt文件读取
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0

# 跑分布式数据
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_CCIX_RING=1


#export BKCL_KL3_SYSCON_FLAG=1 #需要注释
export BKCL_RING_BUFFER_GM=1
export BKCL_FORCE_SYNC=1
export BKCL_CLUSTERS_PER_RING=2 #13b 8卡需要调低, 2卡适当调高, 速度快
export BKCL_RING_BUFFER_SIZE=2048000
export XPUSIM_TOPOLOGY_FILE="ccix_inter.txt"
ulimit -c 0
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=3600


# from paddle
#export BKCL_TREE_THRESHOLD=0
export XPU_ZEBU_MODE=1
export BKCL_XLINK_C2C=1
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_CLUSTERS_PER_RING=2
export BKCL_RING_BUFFER_SIZE=2048000
export XPUSIM_TOPOLOGY_FILE="ccix_inter.txt"

unset BKCL_KL3_SYSCON_FLAG

 #p-mode
export BKCL_TREE_THRESHOLD=1

unset XPU_FORCE_USERMODE_LAUNCH
unset XPU_DUMMY_EVENT

export XMLIR_FA_GEMM_TYPE=float16 #FA GEMM type默认为tf32，较慢，需要指定为fp16


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA2_7B_ARGS="--tensor-model-parallel-size 1 \
                --pipeline-model-parallel-size 1 \
                --num-layers 32 \
                --hidden-size 4096 \
                --ffn-hidden-size 11008 \
                --num-attention-heads 32 \
                --init-method-std 0.01 \
                --micro-batch-size 1 \
                --global-batch-size 1024 \
                --seq-length 4096 \
                --max-position-embeddings 4096 \
                --pad-vocab-size-to 32000 \
                --make-vocab-size-divisible-by 128 \
                --tokenizer-type LLaMASentencePieceTokenizer \
                --tokenizer-model $TOKENIZER_PATH \
                --lr 1.5e-4 \
                --min-lr 1.5e-5 \
                --lr-decay-style cosine \
                --weight-decay 1e-2 \
                --clip-grad 1.0 \
                --adam-beta1 0.9 \
                --adam-beta2 0.95 \
                --train-iters 11920 \
                --lr-decay-iters 10251  \
                --lr-warmup-iters 2000   \
                --rmsnorm-epsilon 1e-6 \
                --activation-func swiglu \
                --use-distributed-optimizer \
                --use-rotary-position-embeddings \
                --apply-rotary-positional-embedding-kernel \
                --untie-embeddings-and-output-weights \
                --attention-dropout 0 \
                --hidden-dropout 0 \
                --embedding-dropout 0 \
                --disable-bias-linear \
                --no-bias-gelu-fusion \
                --no-position-embedding \
                --no-masked-softmax-fusion \
                --no-gradient-accumulation-fusion \
                --attention-softmax-in-fp32 \
                --no-query-key-layer-scaling \
                --initial-loss-scale 16 \
                --memory-saving \
                --fused-rmsnorm \
                --use-flash-attn \
                --use-cpu-initialization \
                --fp16"
                #--fuse-qk-rotary-pos-emb \

OUTPUT_ARGS="--log-interval 1 \
             --detail-log-interval 3 \
             --timing-log-option all \
             --save-interval 500 \
             --eval-interval 50000 \
             --eval-iters 0"
             # --timing-log-level 2 \

OTHER_ARGS="--data-path $DATA_PATH \
            --split 100,0,0 \
            --DDP-impl local \
            --data-impl mmap \
            --load $CHECKPOINT_LOAD_PATH \
            --save $CHECKPOINT_SAVE_PATH \
            --distributed-backend nccl"

XFLAGS --disable megatron_23_05
XFLAGS --disable megatron_aiak
# XFLAGS --disable megatron
# XFLAGS --enable megatron_aiak
PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA2_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
