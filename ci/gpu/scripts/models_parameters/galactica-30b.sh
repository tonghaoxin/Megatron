#! /bin/bash
set -eo pipefail

############################################ Kubernetes 训练任务参数 ############################################
# 要求使用 Kubernetes集群训练必须设置的参数, 含义: 使用K8S 集群训练使用的 Node 总数, 比如可以设置: 4、5、6 个Node 节点用来训练
# 例如:
# 单机场景：1个 Node 节点用来训练
# 多机场景：2个 Node 节点用来训练
TOTAL_K8S_NODES=2

############################################ Kubernetes 训练任务参数 ############################################

############################################ 模型训练参数 ############################################

ROOT_DIR=/workspace/Megatron-LM/examples
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-"/mnt/pfs/leoli"}
DATA_PATH=${TRAIN_DATA_DIR}/datasets/galactica/pile_galactica_test/pile-galactica_text_document
CHECKPOINT_PATH=${TRAIN_DATA_DIR}/megatron_checkpoint/galactica/megatron_galactica_30b_checkpoint_tp2_pp2_dp4_zero1
TOKENIZER_PATH=${TRAIN_DATA_DIR}/tokenizers/galactica/galactica-hf-config

TENSORBOARD_PATH="./tensorborads/galactica-30b"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

TP_SIZE=2
PP_SIZE=2

############################################ 模型训练参数 ############################################

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

GALACTICA_30B_ARGS="--tensor-model-parallel-size $TP_SIZE \
                     --pipeline-model-parallel-size $PP_SIZE \
                     --num-layers 48 \
                     --hidden-size 7168 \
                     --num-attention-heads 56 \
                     --micro-batch-size 2 \
                     --global-batch-size 128 \
                     --seq-length 1024 \
                     --max-position-embeddings 2048 \
                     --lr 1.0e-5 \
                     --min-lr 1.0e-5 \
                     --clip-grad 1.0 \
                     --weight-decay 1e-2 \
                     --train-iters 100 \
                     --lr-decay-iters 320000 \
                     --lr-decay-style linear \
                     --lr-warmup-fraction .01 \
                     --fp16 \
                     --tokenizer-type GalacticaHFTokenizer \
                     --hf-tokenizer-path ${TOKENIZER_PATH} \
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
            --load $CHECKPOINT_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --tensorboard-dir $TENSORBOARD_PATH \
            --distributed-backend nccl"
       #      --save $CHECKPOINT_PATH \

export PYTHONPATH="$ROOT_DIR/..":"$ROOT_DIR/../megatron/accelerator/custom_ops/cuda_ops":$PYTHONPATH

command="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_galactica.py \
       $GALACTICA_30B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS"
echo "执行命令: "$command""
eval $command
