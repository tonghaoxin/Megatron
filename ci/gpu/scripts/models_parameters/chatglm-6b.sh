#! /bin/bash
set -eo pipefail

############################################ Kubernetes 训练任务参数 ############################################
# 要求使用 Kubernetes集群训练必须设置的参数, 含义: 使用K8S 集群训练使用的 Node 总数, 比如可以设置: 4、5、6 个Node 节点用来训练
# 例如:
# 单机场景：1个 Node 节点用来训练
# 多机场景：2个 Node 节点用来训练
TOTAL_K8S_NODES=1

############################################ Kubernetes 训练任务参数 ############################################

############################################ 模型训练参数 ############################################

ROOT_DIR=/workspace/Megatron-LM/examples
TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-"/mnt/pfs/leoli"}
DATA_PATH=${TRAIN_DATA_DIR}/datasets/wudao
CHECKPOINT_PATH=${TRAIN_DATA_DIR}/megatron_checkpoint/chatglm-6b-meg/tp1-pp1-dp8
TOKENIZER_PATH=${TRAIN_DATA_DIR}/tokenizers/chatglm-6b-tokenizers/ice_text.model

TENSORBOARD_PATH=./tensorborads/chatglm-6b

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-1217}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


NLAYERS=28
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=128

TP_SIZE=1
PP_SIZE=1
SEQ_LEN=1024
NHEADS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=16384

############################################ 模型训练参数 ############################################

GLM_ARGS="--num-layers $NLAYERS \
        --hidden-size $HIDDEN_SIZE \
        --ffn-hidden-size $FFN_HIDDEN_SIZE \
        --num-attention-heads $NHEADS \
        --make-vocab-size-divisible-by 768 \
        --positional-encoding-2d \
        --rotary-embedding-2d \
        --model-type-custom chatglm-6b \
        --use-rotary-position-embeddings-with-position-ids \
        --seq2seq-data-collator \
        --deepnorm"

TRAINING_ARGS="--train-iters 100 \
        --lr-decay-iters 32 \
        --lr-warmup-fraction 0.01 \
        --tensor-model-parallel-size $TP_SIZE \
        --pipeline-model-parallel-size $PP_SIZE \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --seq-length $SEQ_LEN \
        --num-workers 1 \
        --use-distributed-optimizer \
        --data-path $DATA_PATH \
        --abort-on-unmet-fused-kernel-constraints \
        --split 949,50,1 \
        --distributed-backend nccl \
        --init-method-std 0.0052 \
        --no-async-tensor-model-parallel-allreduce \
        --no-position-embedding \
        --tokenizer-model ${TOKENIZER_PATH} \
        --initial-loss-scale 16 \
        --loss-scale 0 \
        --hidden-dropout 0 \
        --attention-dropout 0 \
        --embedding-dropout 0 \
        --load $CHECKPOINT_PATH \
        --fp16"
        #   --save $CHECKPOINT_PATH \

OPTIMIZER_ARGS="--optimizer adam \
        --adam-beta1 0.9 \
        --adam-beta2 0.995 \
        --adam-eps 1e-8 \
        --lr 1e-5 \
        --min-lr 1e-6 \
        --override-opt_param-scheduler \
        --lr-decay-style cosine \
        --clip-grad 1.0 \
        --weight-decay 1e-1"

OUTPUT_ARGS="--log-interval 1 \
        --save-interval 10000 \
        --eval-interval 1000 \
        --eval-iters 3 \
        --tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-queue-size 5 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard"

export PYTHONPATH="$ROOT_DIR/..":"$ROOT_DIR/../megatron/accelerator/custom_ops/cuda_ops":$PYTHONPATH

command="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        $ROOT_DIR/../pretrain_chatglm.py \
        $GLM_ARGS \
        $TRAINING_ARGS \
        $OPTIMIZER_ARGS \
        $OUTPUT_ARGS"
echo "执行命令: "$command""
eval $command
