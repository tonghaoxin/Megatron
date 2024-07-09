#! /bin/bash

set -e
set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

# DATA_PATH=$ROOT_DIR/../data/alpaca_text_document
DATA_PATH=$ROOT_DIR/../data/pile-llama_text_document
TOKENIZER_PATH=$ROOT_DIR/../tokenizer/tokenizer.model
CHECKPOINT_LOAD_PATH=$ROOT_DIR/../checkpoints/megatron_llama2_13b_checkpoint_tp1_pp1_dp8_zero1_fp16
CHECKPOINT_SAVE_PATH=$ROOT_DIR/../checkpoints/megatron_llama2_13b_checkpoint_tp1_pp1_dp8_zero1_fp16_saved

export LD_LIBRARY_PATH=$ROOT_DIR/../xccl-so:$ROOT_DIR/../xre-so:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${1:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9919}
NNODES=${2:-1}
NODE_RANK=${3:-0}

ulimit -c 0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA2_13B_ARGS="--tensor-model-parallel-size 1 \
                 --pipeline-model-parallel-size 1 \
                 --num-layers 40 \
                 --hidden-size 5120 \
                 --ffn-hidden-size 13824 \
                 --num-attention-heads 40 \
                 --init-method-std 0.01 \
                 --micro-batch-size 1 \
                 --global-batch-size 1024 \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --pad-vocab-size-to 32000 \
                 --tokenizer-type LLaMASentencePieceTokenizer \
                 --tokenizer-model $TOKENIZER_PATH \
                 --lr 1.5e-4 \
                 --min-lr 1.5e-5 \
                 --lr-decay-style cosine \
                 --weight-decay 1e-2 \
                 --clip-grad 1.0 \
                 --initial-loss-scale 16 \
                 --adam-beta1 0.9 \
                 --adam-beta2 0.95 \
                 --train-iters 50000 \
                 --lr-decay-iters 320000 \
                 --lr-warmup-fraction 0.01 \
                 --sequence-parallel \
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
                 --no-query-key-layer-scaling \
                 --no-gradient-accumulation-fusion \
                 --attention-softmax-in-fp32 \
                 --memory-saving \
                 --fused-rmsnorm \
                 --use-flash-attn \
                 --use-cpu-initialization \
                 --fp16"
                 # --recompute-granularity full \
                 # --recompute-method block \
                 # --recompute-num-layers 1 \
                 # --enable-sequence-parallel-caching \
                 # --fuse-qk-rotary-pos-emb \
                 # --tp-gemm-parallel \
                 # --adam-eps 1e-05 \
                 # --no-async-tensor-model-parallel-allreduce \

OUTPUT_ARGS="--log-interval 1 \
             --detail-log-interval 3 \
             --timing-log-option all \
             --save-interval 50000 \
             --eval-interval 50000 \
             --exit-interval 5 \
             --eval-iters 0"
             # --timing-log-level 2 \

OTHER_ARGS="--data-path $DATA_PATH \
            --split 100,0,0 \
            --DDP-impl local \
            --data-impl mmap \
            --load $CHECKPOINT_LOAD_PATH \
            --save $CHECKPOINT_SAVE_PATH \
            --distributed-backend nccl"


#################################
# export XPURT_DISPATCH_MODE=PROFILING
# export xpu_kernel_debug=true
# export XPUAPI_DEBUG=0x1001

#################################
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
export DIST_MULTI_STREAM=true


# export FAST_SWIGLU_ENABLE=1
export XMLIR_FA_GEMM_TYPE=float16
export XBLAS_FC_HBM_VERSION=40
# export USE_FAST_BF16_FC=true
# export FAST_BF16_FC_WITHOUT_CACHE=true
export XMLIR_PARALLEL_SAVE_MEMORY=true
# export XMLIR_PARALLEL_SAVE_MEMORY=false
# export XDNN_HIGH_ACCRACY_FP32_TO_BF16=true


#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

##############################
export BKCL_RDMA_PROXY_DISABLE=1 # 屏蔽旧架构
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_FLAT_RING=1
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_BUFFER_GM=1
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=1
export XPU_ZEBU_MODE=1
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
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
# export BKCL_FORCE_SYNC=1


#################################
# 13B 模型训练
#################################
XFLAGS --disable megatron_23_05
XFLAGS --disable megatron_aiak
# XFLAGS --disable megatron
# XFLAGS --enable megatron_aiak
PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
  torchrun $DISTRIBUTED_ARGS \
  $ROOT_DIR/../pretrain_llama.py \
  $LLAMA2_13B_ARGS \
  $OUTPUT_ARGS \
  $OTHER_ARGS
