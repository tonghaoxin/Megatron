#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../data/llama_00_text_document
TOKENIZER_PATH=$ROOT_DIR/../tokenizer/tokenizer.model
CHECKPOINT_PATH=${ROOT_DIR}/../checkpoints/
# VOCAB_FILE=${ROOT_DIR}/../data/gpt2-vocab.json
# MERGE_FILE=${ROOT_DIR}/../data/gpt2-merges.txt

GPUS_PER_NODE=8
# Change for multinode config
MASTER_PORT=${MASTER_PORT:-8658}
MASTER_ADDR=${1:-"127.0.0.1"}
NNODES=${2:-1}
NODE_RANK=${3:-0}

#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CUDA_DEVICE_MAX_CONNECTIONS=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
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
                 #--torch-profiler-skip-first-times 0 \
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
                 #--pipeline-model-parallel-partition-method imbalanced \

HETERO_ARGS="--hetero-mode pp \
             --hetero-cross-platform \
             --hetero-pipeline-stages 2 19 19 2 21 21 \
             --hetero-device-types gpu npu \
             --hetero-current-device-type gpu"
             # --hetero-micro-batch-sizes 1 1 1 1 \
             # --hetero-pipeline-stages 2 20 20 2 20 20 \

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 50000 \
             --eval-interval 50000 \
             --detail-log-interval 5 \
             --timing-log-option all \
             --eval-iters 0"

OTHER_ARGS="--data-path $DATA_PATH \
            --split 100,0,0 \
            --DDP-impl local \
            --data-impl mmap \
            --distributed-backend nccl"
            #--save $CHECKPOINT_SAVE_PATH \
            #--load $CHECKPOINT_LOAD_PATH \

pkill -9 python
PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/cuda_ops':$PYTHONPATH \
    python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    $ROOT_DIR/../pretrain_llama.py \
    $LLAMA2_70B_ARGS \
    $HETERO_ARGS \
    $OUTPUT_ARGS \
    $OTHER_ARGS
