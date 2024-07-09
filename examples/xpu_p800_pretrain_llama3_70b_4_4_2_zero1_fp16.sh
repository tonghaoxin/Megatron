#! /bin/bash

set -e
set -x

#export BKCL_DEBUG=1

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

export XMLIR_LOG_PREFIX=node4_llama3_fp16
# export XMLIR_FC_AUTOTUNE_BOOL=false
# export XMLIR_FC_AUTOTUNE_FILE_NAME="fc_tune.txt"

DATA_PATH=${DATA_PATH:-"/workdir/llama3_70b_data/pile_test/pile-llama_text_document"}
echo "DATA_PATH: $DATA_PATH"
# CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/70b_CheckPoint"}
HF_TOKENIZER_PATH=${HF_TOKENIZER_PATH:-"/workdir/llama3_70b_data/Meta-Llama-3-70B/"}
echo "HF_TOKENIZER_PATH: $HF_TOKENIZER_PATH"

export LD_LIBRARY_PATH=${SO_PATH:-"$ROOT_DIR/../xccl-so:$ROOT_DIR/../xre-so:$LD_LIBRARY_PATH"}

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"172.16.0.2"}
MASTER_PORT=${MASTER_PORT:-9919}
#first parameter node set as default 2
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-0}

ulimit -c 0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

#################################
# export XLOG_LEVEL="capture=info,dist=info"
#export XPURT_DISPATCH_MODE=PROFILING
#export xpu_kernel_debug=true
#export XPUAPI_DEBUG=0x1001
#export PYTORCH_NO_CUDA_MEMORY_CACHING=1
#export BKCL_PERF_DEBUG=1
#export BKCL_DUMP=1
#export BKCL_DEBUG_SKIP_RUN=1
#export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA3_70B_ARGS="--tensor-model-parallel-size 4 \
                 --pipeline-model-parallel-size 4 \
                 --num-layers 80 \
                 --tokenizer-type HFTokenizer \
                 --hf-tokenizer-path $HF_TOKENIZER_PATH \
                 --hidden-size 8192 \
                 --ffn-hidden-size 28672 \
                 --num-attention-heads 64 \
                 --init-method-std 0.01 \
                 --micro-batch-size 1 \
                 --global-batch-size 1024 \
                 --seq-length 8192 \
                 --max-position-embeddings 8192 \
                 --lr 5.0e-6 \
                 --min-lr 1.0e-7 \
                 --lr-decay-style cosine \
                 --weight-decay 1e-2 \
                 --clip-grad 1.0 \
                 --initial-loss-scale 16 \
                 --adam-beta1 0.9 \
                 --adam-beta2 0.95 \
                 --train-iters 50000 \
                 --eval-iters 0 \
                 --lr-decay-iters 320000 \
                 --lr-warmup-fraction 0.01 \
                 --sequence-parallel \
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
                 --no-gradient-accumulation-fusion \
                 --multi-query-attention \
                 --multi-query-group-num 8 \
                 --ffn-dim-multiplier 1.3 \
                 --multiple-of 4096 \
                 --use-flash-attn \
                 --fused-rmsnorm \
                 --memory-saving \
                 --use-distributed-optimizer \
                 --use-cpu-initialization \
                 --apply-rotary-positional-embedding-kernel \
                 --fp16"


OUTPUT_ARGS="--log-interval 1 \
             --save-interval 10000 \
             --exit-interval 5 \
             --eval-interval 1000"
             #--timing-log-level 2 \
             #--timing-log-option all \
             #--detail-log-interval 3 \

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"
            #--load $CHECKPOINT_LOAD_PATH \
            #--save $CHECKPOINT_SAVE_PATH \

#################################
export CUDART_DUMMY_REGISTER=1 
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
export DIST_MULTI_STREAM=true


# export FAST_SWIGLU_ENABLE=1
export XMLIR_FA_GEMM_TYPE=float16
export XBLAS_FC_HBM_VERSION=40

#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

##############################
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_BUFFER_GM=1
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=1
export XPU_ZEBU_MODE=1
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"}
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000
unset BKCL_KL3_SYSCON_FLAG
#export BKCL_RING_FORCE_BULK=1
#export BKCL_DEBUG=1
#export BKCL_RING_FORCE_FUSE=1
# export BKCL_FORCE_SYNC=0

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
# 70B 模型训练
#################################
XFLAGS --enable megatron_aiak

PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA3_70B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
