#!/bin/bash
# This example will start serving the 345M model.
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

PP_SIZE=4
TP_SIZE=2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 12070"

CHECKPOINT=/workspace/chatglm-tp2-pp4-dp1 #<Path to checkpoint (e.g /345m)>
TOKEN_MODEL=/root/data/ChatGLM2-6B/ptuning/THUDM/chatglm2-6b/tokenizer.model

export CUDA_DEVICE_MAX_CONNECTIONS=1


MEGATRON_REQUIRED_ARGS="\
       --num-layers 28  \
       --use-distributed-optimizer \
       --hidden-size 4096  \
       --ffn-hidden-size 13696 \
       --multi-query-attention \
       --multi-query-group-num 2 \
       --activation-func swiglu \
       --vocab-size 65024 \
       --pad-vocab-size-to 65024 \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear  \
       --add-qkv-bias true \
       --positional-encoding-2d \
       --rotary-embedding-2d-to-1d \
       --model-type-custom chatglm2-6b \
       --tokenizer-type ChatGLM2Tokenizer \
       --use-rotary-position-embeddings-with-position-ids \
       --no-position-embedding \
       --seq2seq-data-collator \
       --num-attention-heads 32  \
       --tokenizer-model $TOKEN_MODEL \
       --micro-batch-size 1  \
       --seq-length 512  \
       --out-seq-length 1024  \
       --temperature 0.1  \
       --top_p 0.9  \
"

CMD="$ROOT_DIR/../tasks/eval_harness/evaluate_chatglm.py \
    --load $CHECKPOINT\
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE\
    --micro-batch-size 1\
    --intermed_results \
    --results_path /test/test.json \
    --eval_fp32 \
    --task_list boolq \
    $MEGATRON_REQUIRED_ARGS
    "

LAUNCHER="torchrun $DISTRIBUTED_ARGS "
$LAUNCHER $CMD