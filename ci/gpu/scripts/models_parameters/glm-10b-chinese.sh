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
DATA_PATH=${TRAIN_DATA_DIR}/datasets/cc_download_articles
CHECKPOINT_PATH=${TRAIN_DATA_DIR}/megatron_checkpoint/glm-10b-ckpt
TOKENIZER_PATH=${TRAIN_DATA_DIR}/tokenizers/glm-10b-tokenizers/cog-pretrain.model

TENSORBOARD_PATH=./tensorborads/GLM-10B

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

TP_SIZE=1
PP_SIZE=1

# global-batch-size：64，32，16 oom，目前只能跑1和8的组合，我丢；

############################################ 模型训练参数 ############################################

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

GLM_10B_ARGS="--num-layers 48 \
              --hidden-size 4096 \
              --ffn-hidden-size 16384 \
              --num-attention-heads 64 \
              --make-vocab-size-divisible-by 128 \
              --single-span-prob 0.02 \
              --short-seq-prob 0.02 \
              --average-block-length 3 \
              --block-lm \
              --task-mask \
              --bert-prob 0.5 \
              --gap-sentence-prob 0.2 \
              --gap-sentence-ratio 0.15 \
              --avg-block-length 3 \
              --gpt-min-ratio 0.25 \
              --block-mask-prob 0.1 \
              --short-seq-prob 0.02"

OPTIMIZER_ARGS="--optimizer adam \
                --adam-beta1 0.9 \
                --adam-beta2 0.999 \
                --adam-eps 1e-8 \
                --lr 1e-5 \
                --min-lr 4e-6 \
                --override-opt_param-scheduler \
                --lr-decay-style cosine \
                --lr-decay-samples 791015625 \
                --lr-warmup-samples 4394531 \
                --clip-grad 1.0 \
                --weight-decay 1e-1"

OUTPUT_ARGS="--log-interval 1 \
                --save-interval 100000000 \
                --eval-interval 1000 \
                --eval-iters 3 \
                --tensorboard-dir $TENSORBOARD_PATH \
                --tensorboard-queue-size 5 \
                --log-timers-to-tensorboard \
                --log-batch-size-to-tensorboard \
                --log-validation-ppl-to-tensorboard"

TRAINING_ARGS="--tensor-model-parallel-size $TP_SIZE \
                --pipeline-model-parallel-size $PP_SIZE \
                --micro-batch-size 1 \
                --global-batch-size 8 \
                --seq-length 1024 \
                --max-position-embeddings 1025 \
                --abort-on-unmet-fused-kernel-constraints \
                --train-samples 878906250 \
                --split 949,50,1 \
                --distributed-backend nccl \
                --init-method-std 0.0052 \
                --hidden-dropout 0 \
                --attention-dropout 0 \
                --embedding-dropout 0 \
                --no-async-tensor-model-parallel-allreduce \
                --tokenizer-type GLMChineseTokenizer \
                --tokenizer-model $TOKENIZER_PATH \
                --use-distributed-optimizer \
                --no-query-key-layer-scaling \
                --model-type-custom glm-10b \
                --data-name cc-news \
                --dataloader-type cyclic \
                --recompute-granularity full \
                --recompute-method block \
                --recompute-num-layers 8 \
                --initial-loss-scale 16 \
                --min-loss-scale 1 \
                --loss-scale-window 1000 \
                --loss-scale 0 \
                --hysteresis 2
                --fp16"

OTHER_ARGS="--load $CHECKPOINT_PATH \
        --data-path $DATA_PATH"
        # --save $CHECKPOINT_PATH \

export PYTHONPATH="$ROOT_DIR/..":"$ROOT_DIR/../megatron/accelerator/custom_ops/cuda_ops":$PYTHONPATH

command="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
        $ROOT_DIR/../pretrain_glm_10b.py \
        $GLM_10B_ARGS \
        $OPTIMIZER_ARGS \
        $OUTPUT_ARGS \
        $TRAINING_ARGS \
        $OTHER_ARGS"
echo "执行命令: "$command""
eval $command
