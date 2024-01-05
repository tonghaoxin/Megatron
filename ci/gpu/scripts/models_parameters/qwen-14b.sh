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
DATA_PATH=${TRAIN_DATA_DIR}/qwen/qwen-data/qwen_00_text_document_content_sentence
CHECKPOINT_PATH=${TRAIN_DATA_DIR}/qwen/Qwen-14b_tp2pp1dp4
TOKENIZER_PATH=${TRAIN_DATA_DIR}/qwen/Qwen-14b_tp2pp1dp4/qwen.tiktoken

TENSORBOARD_PATH=./tensorborads/qwen-14b

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

TP_SIZE=2
PP_SIZE=1

############################################ 模型训练参数 ############################################

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

QWEN_14B_ARGS="--tensor-model-parallel-size ${TP_SIZE} \
               --pipeline-model-parallel-size ${PP_SIZE} \
               --num-layers 40 \
               --hidden-size 5120 \
               --ffn-hidden-size 13696 \
               --num-attention-heads 40 \
               --micro-batch-size 1 \
               --global-batch-size 128 \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --lr 1.0e-5 \
               --min-lr 1.0e-6 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --weight-decay 0.1 \
               --adam-eps 1e-08 \
               --train-iters 6400000 \
               --lr-decay-iters 6400000 \
               --lr-decay-style linear \
               --lr-warmup-iters 6400 \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-type QwenTokenizer \
               --tokenizer-model ${TOKENIZER_PATH} \
               --initial-loss-scale 16 \
               --swiglu \
               --disable-bias-linear \
               --add-qkv-bias True \
               --use-rotary-position-embeddings \
               --no-position-embedding \
               --untie-embeddings-and-output-weights \
               --use-flash-attn \
               --init-method-std 0.006 \
               --no-load-optim \
               --no-load-rng \
               --num-workers 8 \
               --seed 1234 \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --use-distributed-optimizer \
               --accumulate-allreduce-grads-in-fp32 \
               --bf16"

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--load $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 98,2,0 \
            --distributed-backend nccl"
       #      --save $CHECKPOINT_PATH \

export PYTHONPATH="$ROOT_DIR/..":"$ROOT_DIR/../megatron/accelerator/custom_ops/cuda_ops":$PYTHONPATH

command="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
      $ROOT_DIR/../pretrain_qwen.py \
      $QWEN_14B_ARGS \
      $OUTPUT_ARGS \
      $OTHER_ARGS"
echo "执行命令: "$command""
eval $command
