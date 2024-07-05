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
                 --recompute-granularity full \
                 --recompute-method block \
                 --recompute-num-layers 1 \
                 --fp16"
                 # --enable-sequence-parallel-caching \
                 # --fuse-qk-rotary-pos-emb \
                 # --tp-gemm-parallel \
                 # --adam-eps 1e-05 \
                 # --no-async-tensor-model-parallel-allreduce \

OUTPUT_ARGS="--log-interval 1 \
             --detail-log-interval 3 \
             --timing-log-option all \
             --save-interval 500 \
             --eval-interval 50000 \
             --exit-interval 2 \
             --eval-iters 0"
             # --timing-log-level 2 \

OTHER_ARGS="--data-path $DATA_PATH \
            --split 100,0,0 \
            --DDP-impl local \
            --data-impl mmap \
            --save $CHECKPOINT_SAVE_PATH \
            --load $CHECKPOINT_LOAD_PATH \
            --distributed-backend nccl"

#################################
# DEBUG Ó;²¿·Ö#################################
# export XLOG_LEVEL="capture=info,dist=info"
# export XPURT_DISPATCH_MODE=PROFILING
# export xpu_kernel_debug=true
# export XPUAPI_DEBUG=0x1001

#################################
# driver·½ʽ²¿·Ö#################################
#pmode ģʽÐҪ¿ªÆÏÃ»·¾³±ä
export CUDART_DUMMY_REGISTER=1 #ǿÖ__cudaRegister***·µ»سɹ¦£¬ÓÈ¹ých»ìNV¹ٷ½cublas/cudnnÔ³ɵÄæÐΪ

#imode ģʽÐҪͬʱ¿ªÆÏÁ3¸����³±ä
export CUDART_DUMMY_REGISTER=1 #ǿÖ__cudaRegister***·µ»سɹ¦£¬ÓÈ¹ých»ìNV¹ٷ½cublas/cudnnÔ³ɵÄæÐΪ
export XPU_FORCE_USERMODE_LAUNCH=1 #ǿÖʹÓ´¿Ó»§̬¸ßÔÜaunch/itģʽ£¬¿ÉԻñí×СµÄaunch¿ªÏ£¬ÐҪkunlun.ko insmodʱʹÓ
# export XPU_DUMMY_EVENT=1 #ǿÖeventÏ¹ØPI·µ»سɹ¦£¬µ±ǰÓÓevent»¹δreadyµĴ¿Ó»§̬¸ßÔÜaunch/waitģʽ
export DIST_MULTI_STREAM=true

#################################
# Ë×²¿·Ö#################################
# export FAST_SWIGLU_ENABLE=1 # ²éSWIGLUË×ʵÏ
export XMLIR_FA_GEMM_TYPE=float16 # FA GEMM typeĬÈΪtf32£¬½Ïýªָ¶¨Ϊfp16
# export XPUAPI_DEFAULT_SIZE=4000000000 # ¼õãÖÐԤ·Öä´浼ÖµÄXPU_WAIT¿ªÏ, ¸±×ÓÊ¶î¶àºÄԴæxport XBLAS_FC_HBM_VERSION=40
# export XMLIR_API_DEFAULT_SIZE=3000000000

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
export BKCL_TREE_THRESHOLD=1
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
#export BKCL_RING_FORCE_BULK=1
#export BKCL_DEBUG=1
#export BKCL_RING_FORCE_FUSE=1

export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=1
export XPU_ZEBU_MODE=1 # ӰÏccix_inter.txtÎ¼þ¶Á¡
export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_RING_BUFFER_GM=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152
export XPUSIM_TOPOLOGY_FILE="ccix_inter.txt"
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

#python test_clone.py
#################################
# 13B 模型训练
#################################
# XFLAGS --disable megatron
# XFLAGS --enable megatron_aiak
PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH \
  torchrun $DISTRIBUTED_ARGS \
  $ROOT_DIR/../pretrain_llama.py \
  $LLAMA2_13B_ARGS \
  $OUTPUT_ARGS \
  $OTHER_ARGS
