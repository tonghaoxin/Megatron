#! /bin/bash

set -e
set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )


################################# 
# config
export XMLIR_LOG_PREFIX=node4
export XMLIR_FC_AUTOTUNE_BOOL=false
# export XMLIR_FC_AUTOTUNE_FILE_NAME="fc_tune.txt"

DATA_PATH=${DATA_PATH:-"$ROOT_DIR/../data/pile-llama_text_document"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/70b_CheckPoint"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"$ROOT_DIR/../tokenizer/tokenizer.model"}

export LD_LIBRARY_PATH=${SO_PATH:-"$ROOT_DIR/../xccl-so:$ROOT_DIR/../xre-so:$LD_LIBRARY_PATH"}

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"172.16.0.2"}
MASTER_PORT=${MASTER_PORT:-9919}
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-0}

PP_SIZE=${PP_SIZE:-4}
TP_SIZE=${TP_SIZE:-4}
NUM_LAYERS=${NUM_LAYERS:-80}
GLOBAL_BS=${GLOBAL_BS:-1024}
MICRO_BS=${MICRO_BS:-2}
EXIT_ITER=${EXIT_ITER:-2}
LOSS_SCALE=${LOSS_SCALE:-16} 

ulimit -c 0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

################################# 
# Debugging
# export XLOG_LEVEL="capture=info,dist=info"
# export XPURT_DISPATCH_MODE=PROFILING
# export XPUAPI_DEBUG=0x1
# export BKCL_DEBUG=1
#export BKCL_PERF_DEBUG=1
#export BKCL_DUMP=1
#export BKCL_DEBUG_SKIP_RUN=1
#export PYTORCH_NO_CUDA_MEMORY_CACHING=1



DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"


LLAMA2_70B_ARGS="--tensor-model-parallel-size ${TP_SIZE} \
                 --pipeline-model-parallel-size ${PP_SIZE} \
                 --num-layers ${NUM_LAYERS} \
                 --hidden-size 8192 \
                 --ffn-hidden-size 28672 \
                 --num-attention-heads 64 \
                 --init-method-std 0.01 \
                 --micro-batch-size ${MICRO_BS} \
                 --global-batch-size ${GLOBAL_BS} \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --tokenizer-type LLaMASentencePieceTokenizer \
                 --tokenizer-model $TOKENIZER_PATH \
                 --lr 5.0e-6 \
                 --min-lr 1.0e-7 \
                 --lr-decay-style cosine \
                 --weight-decay 1e-2 \
                 --clip-grad 1.0 \
                 --initial-loss-scale ${LOSS_SCALE} \
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
                 # --cpu-offload manual \
                 # --cpu-offload-percent 1 \
                 # --num-layers-per-virtual-pipeline-stage 2 \
                 # --pipeline-model-parallel-partition-method imbalanced \


OUTPUT_ARGS="--log-interval 1 \
             --save-interval 5000 \
             --exit-interval ${EXIT_ITER} \
             --eval-interval 1000"
             #--timing-log-level 2 \
             # --timing-log-option all \
             # --detail-log-interval 4 \

OTHER_ARGS="--data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --save $CHECKPOINT_PATH \
            --distributed-backend nccl"
            #--load $CHECKPOINT_LOAD_PATH \

#################################
# Runtime
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
# export XPU_DUMMY_EVENT=1

################################# 
# XMLIR
export DIST_MULTI_STREAM=true
export XMLIR_DIST_SINGLETON_STREAM=true
# export FAST_SWIGLU_ENABLE=1
export XMLIR_FA_GEMM_TYPE=float16
# export XPUAPI_DEFAULT_SIZE=2000000000
# export XMLIR_API_DEFAULT_SIZE=1000000000
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # disallow cpu fallback
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #  disallow cpu fallback
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

################################# 
# BKCL
export BKCL_CCIX_RING=1
export BKCL_CCIX_BUFFER_GM=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_BUFFER_GM=1
# export BKCL_RING_FORCE_BULK=1
#export BKCL_DEBUG=1
#export BKCL_RING_FORCE_FUSE=1
export BKCL_RDMA_PROXY_DISABLE=1

export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=0
export XPU_ZEBU_MODE=1

#depends on the network card
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"}
# export BKCL_RDMA_NICS=ens11np0,ens12np0,ens13np0,ens14np0,ens15np0,ens16np0,ens17np0,ens18np0
#export BKCL_RDMA_NICS=xgbe1,xgbe1,xgbe2,xgbe2,xgbe3,xgbe3,xgbe4,xgbe4
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_RING_BUFFER_GM=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000
unset BKCL_KL3_SYSCON_FLAG


################################# 
# timeout
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
XFLAGS --disable megatron_23_05
XFLAGS --enable megatron_aiak
# XFLAGS --disable megatron
# XFLAGS --enable megatron_aiak
PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA2_70B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
