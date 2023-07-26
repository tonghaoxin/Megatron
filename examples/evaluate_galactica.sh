#!/bin/bash
# This example will start serving the 345M model.
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
HF_TOKENIZER_PATH=/ssd2/galactic/galactica-hf-config

PP_SIZE=1
TP_SIZE=8
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 12070"

CHECKPOINT=/workspace/galactica-6.7b-tp8-pp1-dp1 #<Path to checkpoint (e.g /345m)>

export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_REQUIRED_ARGS="\
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 16384 \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --use-learned-position-embeddings \
       --disable-bias-linear \
       --hf-tokenizer-path ${HF_TOKENIZER_PATH} \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --out-seq-length 1024  \
       --temperature 0.1  \
       --top_p 0.9  \
"

CMD="$ROOT_DIR/../tasks/eval_harness/evaluate_galactica.py \
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
进入IDE阅读
