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
       --num-layers 32  \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --disable-bias-linear \
       --num-attention-heads 32  \
       --max-position-embeddings 2048  \
       --no-position-embedding \
       --tokenizer-type LLaMASentencePieceTokenizer \
       --tokenizer-model $TOKENIZER_PATH/tokenizer.model \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --out-seq-length 1024  \
       --temperature 0.1  \
       --top_p 0.9  \
       --swiglu \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --seed 42"
       # --use-distributed-optimizer \

CMD="PYTHONPATH=/workspace/lm-evaluation-harness:$PYTHONPATH \
    $ROOT_DIR/../tasks/eval_harness/evaluate_llama.py \
    --load $CHECKPOINT_LOAD_PATH \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE\
    --micro-batch-size 1\
    --intermed_results\
    --results_path $ROOT_DIR/../eval_llama.json \
    --eval_fp32\
    --task_list boolq\
    $MEGATRON_REQUIRED_ARGS
    "

LAUNCHER="torchrun $DISTRIBUTED_ARGS "
$LAUNCHER $CMD
