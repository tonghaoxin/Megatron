#! /bin/bash

set -e
set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

export XMLIR_LOG_PREFIX=node4
export XMLIR_FC_AUTOTUNE_BOOL=false
export XMLIR_FC_AUTOTUNE_FILE_NAME="fc_tune.txt"

# DATA_PATH=/workspace/70b_datasets/alpaca/alpaca_text_document
# DATA_PATH=$ROOT_DIR/../data/alpaca_text_document
DATA_PATH=${DATA_PATH:-"$ROOT_DIR/../data/pile-llama_text_document"}
# CHECKPOINT_LOAD_PATH=/home/jiangyu/megatron_datasets/llama/checkpoints/megatron_llama_13b_checkpoint_tp8_pp1_fp32
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/workspace/70b_CheckPoint"}

# TOKENIZER_PATH=/workspace/megatron_datasets/llama/tokenizer_model/tokenizer.model
TOKENIZER_PATH=${TOKENIZER_PATH:-"$ROOT_DIR/../tokenizer/tokenizer.model"}
# CHECKPOINT_PATH=/workspace/70b_CheckPoint

#export LD_LIBRARY_PATH=$ROOT_DIR/../xccl-so:$ROOT_DIR/../xre-so:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/workspace/70b_bkcl_so/0429:/workspace/xre-Linux-x86_64-0.0.0.1-2024-04-16-23-03-29-daily/so:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${SO_PATH:-"$ROOT_DIR/../xccl-so:$ROOT_DIR/../xre-so:$LD_LIBRARY_PATH"}
# export LD_LIBRARY_PATH=/workspace/output/xccl_rdma-ubuntu_x86_64/so:/workspace/xre-Linux-x86_64-0.0.0.1-2024-04-16-23-03-29-daily/so:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"172.16.0.2"}
MASTER_PORT=${MASTER_PORT:-9919}
#first parameter node set as default 2
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:-0}

PP_SIZE=${PP_SIZE:-4}
TP_SIZE=${TP_SIZE:-4}
NUM_LAYERS=${NUM_LAYERS:-80}
GLOBAL_BS=${GLOBAL_BS:-1024}
MICRO_BS=${MICRO_BS:-2}

ulimit -c 0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

#################################
##################################
# export XLOG_LEVEL="capture=info,dist=info"
# export XPURT_DISPATCH_MODE=PROFILING
# export XPUAPI_DEBUG=0x1

#export PYTORCH_NO_CUDA_MEMORY_CACHING=1

#export BKCL_PERF_DEBUG=1
#export BKCL_DEBUG=1
#export BKCL_DUMP=1
#export BKCL_DEBUG_SKIP_RUN=1

#export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

                 #--pipeline-model-parallel-size 2 \
                 #--num-layers 40 \
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
                 # --cpu-offload manual \
                 # --cpu-offload-percent 1 \
                 # --num-layers-per-virtual-pipeline-stage 2 \
                 # --pipeline-model-parallel-partition-method imbalanced \


OUTPUT_ARGS="--log-interval 1 \
             --save-interval 5000 \
             --exit-interval 2 \
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
# driver·½ʽ²¿·Ö#################################
#pmode ģʽÐҪ¿ªÆÏÃ»·¾³±ä
export CUDART_DUMMY_REGISTER=1 #ǿÖ__cudaRegister***·µ»سɹ¦£¬ÓÈ¹ých»ìNV¹ٷ½cublas/cudnnÔ³ɵÄæÐΪ

#imode ģʽÐҪͬʱ¿ªÆÏÁ3¸����³±ä
export CUDART_DUMMY_REGISTER=1 #ǿÖ__cudaRegister***·µ»سɹ¦£¬ÓÈ¹ých»ìNV¹ٷ½cublas/cudnnÔ³ɵÄæÐΪ
export XPU_FORCE_USERMODE_LAUNCH=1 #ǿÖʹÓ´¿Ó»§̬¸ßÔÜaunch/itģʽ£¬¿ÉԻñí×СµÄaunch¿ªÏ£¬ÐҪkunlun.ko insmodʱʹÓ
# export XPU_DUMMY_EVENT=1 #ǿÖeventÏ¹ØPI·µ»سɹ¦£¬µ±ǰÓÓevent»¹δreadyµĴ¿Ó»§̬¸ßÔÜaunch/waitģʽ
export DIST_MULTI_STREAM=true
export XMLIR_DIST_SINGLETON_STREAM=true

#################################
# Ë×²¿·Ö#################################
# export FAST_SWIGLU_ENABLE=1 # ²éSWIGLUË×ʵÏ
export XMLIR_FA_GEMM_TYPE=float16 # FA GEMM typeĬÈΪtf32£¬½Ïýªָ¶¨Ϊfp16
# export XPUAPI_DEFAULT_SIZE=2000000000 # ¼õãÖÐԤ·Öä´浼ÖµÄXPU_WAIT¿ªÏ, ¸±×ÓÊ¶î¶àºÄԴæxport XBLAS_FC_HBM_VERSION=40
# export XMLIR_API_DEFAULT_SIZE=1000000000 # ¼õãÖÐԤ·Öä´浼ÖµÄXPU_WAIT¿ªÏ, ¸±×ÓÊ¶î¶àºÄԴæxport XBLAS_FC_HBM_VERSION=40

#################################
# Ë×¼ì²¿·Ö#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0 # ½ûNN³öólback CPU
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False #½ûºÎallback
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

#################################
# hbm²¿·Ö#################################
# export XPU_FORCE_CODE_PARAM_LOCATE_IN_L3=1 #¸ßµhbm±ؿª, stable/lowѡ¿ª

#################################
# BKCL C2C²¿·Ö#################################
export BKCL_CCIX_RING=1
export BKCL_CCIX_BUFFER_GM=1

# ccix_inner_8chips

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


# cat > ccix_inter.txt <<EOF
# [chip 0, port 0] <===> [chip 6, port 3]
# [chip 4, port 1] <===> [chip 5, port 2]
# [chip 2, port 3] <===> [chip 4, port 0]
# [chip 2, port 1] <===> [chip 3, port 2]
# [chip 1, port 1] <===> [chip 3, port 1]
# [chip 0, port 1] <===> [chip 1, port 2]
# [chip 0, port 2] <===> [chip 2, port 2]
# [chip 0, port 3] <===> [chip 3, port 3]
# [chip 1, port 0] <===> [chip 2, port 0]
# [chip 1, port 3] <===> [chip 7, port 0]
# [chip 5, port 1] <===> [chip 7, port 1]
# [chip 3, port 0] <===> [chip 5, port 3]
# [chip 4, port 3] <===> [chip 7, port 3]
# [chip 4, port 2] <===> [chip 6, port 2]
# [chip 6, port 1] <===> [chip 7, port 2]
# [chip 5, port 0] <===> [chip 6, port 0]
# EOF

export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_BUFFER_GM=1
# export BKCL_RING_FORCE_BULK=1
#export BKCL_DEBUG=1
#export BKCL_RING_FORCE_FUSE=1
export BKCL_RDMA_PROXY_DISABLE=1

export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=0
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"}
export XPU_ZEBU_MODE=1 # ӰÏccix_inter.txtÎ¼þ¶Á¡
#depends on the network card
# export BKCL_RDMA_NICS=ens11np0,ens12np0,ens13np0,ens14np0,ens15np0,ens16np0,ens17np0,ens18np0
#export BKCL_RDMA_NICS=xgbe1,xgbe1,xgbe2,xgbe2,xgbe3,xgbe3,xgbe4,xgbe4
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_RING_BUFFER_GM=1
# export BKCL_FORCE_SYNC=1
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
