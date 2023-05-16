#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=/ssd2/datasets/pile_gala7b_test/my-pile_text_document

CHECKPOINT_PATH=/ssd2/models/megatron_galactica_7b_checkpoint_tp2_pp2

HF_TOKENIZER_PATH=/models/galactica-30b/

# tensorboard
TENSORBOARD_DIR="/tensorboard/galactica"
mkdir -p ${TENSORBOARD_DIR}

GPUS_PER_NODE=8

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${1:-1}
NODE_RANK=${2:-0}

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

GALACTICA_7B_ARGS="--tensor-model-parallel-size 1 \
                   --pipeline-model-parallel-size 1 \
                   --num-layers 32 \
                   --hidden-size 4096 \
                   --num-attention-heads 32 \
                   --micro-batch-size 4 \
                   --global-batch-size 128 \
                   --seq-length 1024 \
                   --max-position-embeddings 1024 \
                   --lr 0.0003 \
                   --min-lr 1.0e-5 \
                   --clip-grad 1.0 \
                   --optimizer adam \
                   --adam-beta1 0.9 \
                   --adam-beta2 0.95 \
                   --weight-decay 0.1 \
                   --adam-eps 1e-08 \
                   --train-iters 500000 \
                   --lr-decay-iters 320000 \
                   --lr-decay-style linear \
                   --lr-warmup-fraction .01 \
                   --recompute-method uniform \
                   --no-async-tensor-model-parallel-allreduce \
                   --fp16
                   --tokenizer-type GalacticaHFTokenizer \
                   --hf-tokenizer-path ${HF_TOKENIZER_PATH} \
                   --attention-dropout 0.1 \
                   --hidden-dropout 0.1 \
                   --embedding-dropout 0 \
                   --use-learned-position-embeddings \
                   --disable-bias-linear \
                   --use-distributed-optimizer \
                   --use-flash-attn"
                   # --sequence-parallel

GALACTICA_30B_ARGS="--tensor-model-parallel-size 2 \
                    --pipeline-model-parallel-size 2 \
                    --num-layers 48 \
                    --hidden-size 7168 \
                    --num-attention-heads 56 \
                    --micro-batch-size 2 \
                    --global-batch-size 128 \
                    --seq-length 1024 \
                    --max-position-embeddings 1024 \
                    --lr 0.0003 \
                    --min-lr 1.0e-5 \
                    --clip-grad 1.0 \
                    --optimizer adam \
                    --adam-beta1 0.9 \
                    --adam-beta2 0.95 \
                    --weight-decay 1e-2 \
                    --adam-eps 1e-08 \
                    --train-iters 500000 \
                    --lr-decay-iters 320000 \
                    --lr-decay-style linear \
                    --lr-warmup-fraction .01 \
                    --recompute-method uniform \
                    --no-async-tensor-model-parallel-allreduce \
                    --fp16
                    --tokenizer-type GalacticaHFTokenizer \
                    --hf-tokenizer-path ${HF_TOKENIZER_PATH} \
                    --attention-dropout 0.1 \
                    --hidden-dropout 0.1 \
                    --embedding-dropout 0 \
                    --use-learned-position-embeddings \
                    --disable-bias-linear \
                    --use-distributed-optimizer \
                    --use-flash-attn"
                    # --sequence-parallel

GALACTICA_120B_ARGS="--tensor-model-parallel-size 4 \
                     --pipeline-model-parallel-size 6 \
                     --num-layers 96 \
                     --hidden-size 10240 \
                     --num-attention-heads 80 \
                     --micro-batch-size 2 \
                     --global-batch-size 256 \
                     --seq-length 1024 \
                     --max-position-embeddings 1024 \
                     --lr 0.00015 \
                     --min-lr 1.0e-5 \
                     --clip-grad 1.0 \
                     --optimizer adam \
                     --adam-beta1 0.9 \
                     --adam-beta2 0.95 \
                     --weight-decay 1e-2 \
                     --adam-eps 1e-08 \
                     --train-iters 500000 \
                     --lr-decay-iters 320000 \
                     --lr-decay-style linear \
                     --lr-warmup-fraction .01 \
                     --recompute-method uniform \
                     --no-async-tensor-model-parallel-allreduce \
                     --fp16
                     --tokenizer-type GalacticaHFTokenizer \
                     --hf-tokenizer-path ${HF_TOKENIZER_PATH} \
                     --attention-dropout 0.1 \
                     --hidden-dropout 0.1 \
                     --embedding-dropout 0 \
                     --use-learned-position-embeddings \
                     --disable-bias-linear \
                     --use-distributed-optimizer \
                     --use-flash-attn"
                     # --sequence-parallel

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 50000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--data-path $DATA_PATH \
            --save $CHECKPOINT_PATH \
            --load $CHECKPOINT_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl \
            --tensorboard-dir ${TENSORBOARD_DIR}"

PYTHONPATH="$ROOT_DIR/..":$PYTHONPATH \
       python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_galactica.py \
       $GALACTICA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
