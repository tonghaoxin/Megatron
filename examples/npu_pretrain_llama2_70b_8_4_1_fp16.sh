!/bin/bash

set -x

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=$PATH:/opt/rh/devtoolset-7/root/usr/bin
export LD_LIBRARY_PATH=/usr/local/python3.8.18/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.8.18/bin:$PATH
export PATH=/usr/local/python3.8.18/bin/pip:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

DATA_PATH=${ROOT_DIR}/../data/alpaca_text_document
TOKENIZER_PATH=${ROOT_DIR}/../tokenizer/tokenizer.model
CHECKPOINT_LOAD_PATH=${ROOT_DIR}/../checkpoints/megatron_llama2_70b_checkpoint_tp8_pp4_dp1_zero1_fp16
CHECKPOINT_SAVE_PATH=${ROOT_DIR}/../checkpoints/megatron_llama2_70b_checkpoint_tp8_pp4_dp1_zero1_fp16_save
TENSORBOARD_PATH='./llama2_70b_tp8_pp4_dp1_zero1_fp16_tensorboard'

export NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
#export NPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NPUS_PER_NODE=`echo "$NPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
#MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-8658}
MASTER_ADDR=${1:-"127.0.0.1"}
NNODES=${2:-1}
NODE_RANK=${3:-0}

# export OMP_NUM_THREADS=96
# export ASCEND_GLOBAL_LOG_LEVEL=3
# export ASCEND_GLOBAL_EVENT_ENABLE=1
# export INF_NAN_MODE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=1200
# export HCCL_INTRA_PCIE_ENABLE=0
# export HCCL_INTRA_ROCE_ENABLE=1
# export HCCL_RDMA_TC=128
# export HCCL_RDMA_SL=4
export COMBINED_ENABLE=1
# export HCCL_SOCKET_IFNAME=xgbe0
export MULTI_STREAM_MEMORY_REUSE=1

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA2_70B_ARGS="--tensor-model-parallel-size 8 \
                 --pipeline-model-parallel-size 4 \
        	       --num-layers 80 \
        	       --hidden-size 8192 \
        	       --ffn-hidden-size 28672 \
        	       --num-attention-heads 64 \
                 --init-method-std 0.01 \
        	       --micro-batch-size 1 \
        	       --global-batch-size 1024 \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --pad-vocab-size-to 32000 \
                 --tokenizer-type LLaMASentencePieceTokenizer \
                 --tokenizer-model $TOKENIZER_PATH \
        	       --lr 5.0e-5 \
        	       --min-lr 1.0e-6 \
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
                 --fused-rmsnorm \
                 --use-flash-attn \
                 --apply-rotary-positional-embedding-kernel \
                 --use-distributed-optimizer \
		             --memory-saving \
                 --fp16"
                 #--enable-torch-profiler \
                 #--torch-profiler-log-path ./result_dir \
                 #--torch-profiler-wait-times 1 \
                 #--torch-profiler-warmup-times 1 \
                 #--torch-profiler-active-times 1 \
                 #--torch-profiler-repeat-times 1 \
                 #--torch-profiler-skip-first-times 1 \
                 #--recompute-granularity full \
                 #--recompute-method block \
                 #--recompute-num-layers 2 \
		             #--enable-debug \
		             #--debug-layer-list 1 \
		             #--debug-tp-list 0 \
		             #--debug-pp-list 0 \
		             #--debug-dp-list 0 \
                 #--reset-attention-mask \
                 #--make-vocab-size-divisible-by 128 \
                 #--no-async-tensor-model-parallel-allreduce \

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 50000 \
             --eval-interval 50000 \
             --eval-iters 0"

OTHER_ARGS="--data-path $DATA_PATH \
            --split 100,0,0 \
            --DDP-impl local \
            --data-impl mmap \
            --distributed-backend nccl"
            #--save $CHECKPOINT_SAVE_PATH \
            #--load $CHECKPOINT_LOAD_PATH \

PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/npu_ops':$PYTHONPATH \
    python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    $ROOT_DIR/../pretrain_llama.py \
    $LLAMA2_70B_ARGS \
    $OUTPUT_ARGS \
    $OTHER_ARGS
