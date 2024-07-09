#! /bin/bash

#set -e
#set -x

#export BKCL_DEBUG=1

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

export XMLIR_LOG_PREFIX=node8_llama3
export XMLIR_FC_AUTOTUNE_BOOL=false
export XMLIR_FC_AUTOTUNE_FILE_NAME="fc_tune.txt"

#DATA_PATH=/aiak/changtao02/aiak-kunlun/llama2_dataset/pile_llama_test/pile-llama_text_document
DATA_PATH=/aiak/changtao02/aiak-kunlun/llama2_dataset/pile_test/pile-llama_text_document
#CHECKPOINT_LOAD_PATH=/aiak/changtao02/aiak-kunlun/llama2_models/megatron_llama3_70b_checkpoint_tp4_pp4_zero1_fp16
CHECKPOINT_LOAD_PATH=/aiak/changtao02/aiak-kunlun/llama2_models/megatron_llama3_70b_checkpoint_tp4_pp4_zero1_bf16
#CHECKPOINT_LOAD_PATH=/aiak/changtao02/aiak-kunlun/llama2_models/megatron_llama3_70b_checkpoint_tp8_pp4_zero1_fp16
#CHECKPOINT_SAVE_PATH=/aiak/changtao02/aiak-kunlun/llama2_models/megatron_llama3_70b_checkpoint_tp4_pp4_zero1_fp16_saved
CHECKPOINT_SAVE_PATH=/aiak/changtao02/aiak-kunlun/llama2_models/megatron_llama3_70b_checkpoint_tp4_pp4_zero1_bf16_saved
HF_TOKENIZER_PATH=/aiak/changtao02/aiak-kunlun/llama3_70b/Meta-Llama-3-70B/

#export LD_LIBRARY_PATH=/aiak/changtao02/aiak-kunlun/xre-Linux-x86_64-0.0.0.1-2024-04-26-00-05-07-daily/so:/aiak/changtao02/aiak-kunlun/xccl-0506:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/aiak/xccl/xre-Linux-x86_64-5.0.3.1/so:/aiak/xccl/xccl_release/xccl_0524/xccl_999.0.299.3_db3913/so:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/aiak/changtao02/aiak-kunlun/xre-Linux-x86_64-5.0.3.1/so:/aiak/changtao02/aiak-kunlun/xinxuan_fix_bkcl:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/aiak/changtao02/aiak-kunlun/xre-Linux-x86_64-5.0.3.1/so:/aiak/changtao02/aiak-kunlun/xccl_0613:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/aiak/changtao01/aiak-kunlun/xre-Linux-x86_64-5.0.3.1/so:/aiak/wuxinxuan/xccl/xccl_release/xinxuan_2dphangfix_shmfix:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/aiak/changtao02/aiak-kunlun/xre-Linux-x86_64-5.0.3.1/so:/aiak/wuxinxuan/xccl/xccl_release/xccl_0530/xccl_rdma-ubuntu_x86_64_30a4436/so:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/aiak/changtao02/aiak-kunlun/xre-Linux-x86_64-5.0.9.1/so:/aiak/wuxinxuan/xccl/xccl_release/xccl_0624/xccl_rdma_ubuntu_x86_64_c285a77/so:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/aiak/changtao02/aiak-kunlun/xre-Linux-x86_64-5.0.9.1/so:/aiak/wuxinxuan/xccl/xccl_release/xccl_0626/xccl_rdma-ubuntu_x86_64_d387298/so:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/aiak/changtao02/aiak-kunlun/xre-Linux-x86_64-5.0.11.1/so:/aiak/changtao02/aiak-kunlun/xccl_0630:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/aiak/changtao02/aiak-kunlun/xre-Linux-x86_64-5.0.9.1/so:/aiak/changtao02/aiak-kunlun/xccl_0630:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${1:-"172.16.0.2"}
MASTER_PORT=${MASTER_PORT:-9919}
#first parameter node set as default 2
NNODES=${2:-4}
NODE_RANK=${3:-0}

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
                 --lr 0.0002 \
                 --min-lr 1.0e-5 \
                 --lr-decay-style cosine \
                 --weight-decay 1e-2 \
                 --clip-grad 1.0 \
                 --initial-loss-scale 16 \
                 --adam-beta1 0.9 \
                 --adam-beta2 0.95 \
                 --adam-eps 1e-05 \
                 --train-iters 50000 \
                 --eval-iters 0 \
                 --lr-decay-iters 50000 \
                 --lr-warmup-fraction 0.002 \
                 --sequence-parallel \
                 --rmsnorm-epsilon 1e-5 \
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
                 --use-flash-attn \
                 --memory-saving \
                 --use-distributed-optimizer \
                 --fused-rmsnorm \
                 --rotary-emb-base 500000 \
                 --apply-rope-native \
                 --apply-rotary-positional-embedding-kernel \
                 --use-cpu-initialization \
                 --bf16"
                 #--recompute-granularity full \
                 #--recompute-method block \
                 #--recompute-num-layers 1 \
                 #--num-layers-per-virtual-pipeline-stage 5 \
                 #--overlap-p2p-communication \
                 #--no-batch-p2p-communication \
                 #--fp16"
		             #--enable-sequence-parallel-caching \
		             #--distributed-timeout-minutes 30 \
		             #--seed 1234 \

OUTPUT_ARGS="--log-interval 1 \
             --exit-interval 100 \
             --save-interval 10000000 \
             --eval-interval 10000000"
             #--timing-log-level 2 \
             #--timing-log-option all \
             #--detail-log-interval 3 \

OTHER_ARGS="--data-path $DATA_PATH \
            --split 100,0,0 \
            --data-impl mmap \
            --load $CHECKPOINT_LOAD_PATH \
	          --no-load-optim \
            --distributed-backend nccl"
            #--save $CHECKPOINT_SAVE_PATH \
	          #--enable-accelerate-checkpoint \

#################################
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
export DIST_MULTI_STREAM=true
export XMLIR_DIST_SINGLETON_STREAM=true


# export FAST_SWIGLU_ENABLE=1
#export XMLIR_FA_GEMM_TYPE=float16
export XBLAS_FC_HBM_VERSION=40
export USE_FAST_BF16_FC=true
export FAST_BF16_FC_WITHOUT_CACHE=true
#export XMLIR_PARALLEL_SAVE_MEMORY=true
export XMLIR_PARALLEL_SAVE_MEMORY=false
export XDNN_HIGH_ACCRACY_FP32_TO_BF16=true


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
#export BKCL_FORCE_SYNC=1


#################################
# 70B 模型训练
#################################
XFLAGS --disable megatron_23_05
#XFLAGS --disable megatron_aiak
XFLAGS --enable megatron_aiak

PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
       torchrun $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_llama.py \
       $LLAMA3_70B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
