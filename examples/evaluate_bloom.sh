#!/bin/bash
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

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

CHECKPOINT_LOAD_PATH=$ROOT_DIR/../checkpoints
TOKENIZER_PATH=$ROOT_DIR/../tokenizer

export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_REQUIRED_ARGS="
       --num-layers 30 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --micro-batch-size 4 \
       --global-batch-size 64 \
       --max-position-embeddings 512 \
       --no-position-embedding \
       --no-query-key-layer-scaling \
       --tokenizer-type BloomHFTokenizer \
       --hf-tokenizer-path $TOKENIZER_PATH \
       --bf16  \
       --micro-batch-size 4  \
       --seq-length 512 \
       --out-seq-length 512 \
       --temperature 0.1  \
       --top_p 0.9  \
       --use-alibi-position-embeddings \
       --activation-func openai_gelu \
       --pad-vocab-size-to 250880 \
       --add-embed-layernorm \
       --seed 42"
       # --use-distributed-optimizer \
       # --disable-bias-linear \

CMD="$ROOT_DIR/../tasks/eval_harness/evaluate_bloom.py \
     --load $CHECKPOINT_LOAD_PATH \
     --tensor-model-parallel-size $TP_SIZE \
     --pipeline-model-parallel-size $PP_SIZE \
     --intermed_results \
     --results_path $ROOT_DIR/../eval_bloom.json \
     --eval_fp32 \
     --task_list boolq \
     $MEGATRON_REQUIRED_ARGS"

LAUNCHER="torchrun $DISTRIBUTED_ARGS "
$LAUNCHER $CMD
