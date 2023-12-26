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
DATA_PATH=${TRAIN_DATA_DIR}/datasets/aquilachat-7b/convo_samples.jsonl
CHECKPOINT_PATH=${TRAIN_DATA_DIR}/megatron_checkpoint/aquilaChat-7b-tp8-pp1-dp1-fp32
TOKENIZER_PATH=${TRAIN_DATA_DIR}/tokenizers/aquilachat-7b

TENSORBOARD_PATH="./tensorborads/aquila-7b"

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-1217}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

TP_SIZE=8
PP_SIZE=1


AQUILA_7B_ARGS="--tensor-model-parallel-size ${TP_SIZE} \
               --pipeline-model-parallel-size ${PP_SIZE} \
               --num-layers 32 \
               --num-attention-heads 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --micro-batch-size 2 \
               --global-batch-size 128 \
               --num-workers 1 \
               --seq-length 2048 \
               --max-position-embeddings 2048 \
               --pad-vocab-size-to 100008 \
               --lr 1e-5 \
               --min-lr 1.0e-5 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.98 \
               --weight-decay 0.1 \
               --adam-eps 1e-06 \
               --train-iters 100 \
               --lr-decay-iters 80 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .01 \
               --make-vocab-size-divisible-by 8 \
               --no-async-tensor-model-parallel-allreduce \
               --tokenizer-type AQUILABPETokenizer \
               --rmsnorm-epsilon 1e-5 \
               --no-position-embedding \
               --no-gradient-accumulation-fusion \
               --no-bias-dropout-fusion \
               --no-bias-gelu-fusion \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --hf-tokenizer-path ${TOKENIZER_PATH} \
               --loss-scale 0 \
               --initial-loss-scale 16 \
               --attention-softmax-in-fp32 \
               --untie-embeddings-and-output-weights \
               --tensorboard-dir $TENSORBOARD_PATH \
               --tensorboard-queue-size 5 \
               --log-timers-to-tensorboard \
               --log-batch-size-to-tensorboard \
               --swiglu \
               --sequence-parallel \
               --freqs-type complex \
               --no-masked-softmax-fusion \
               --use-rotary-position-embeddings \
               --no-query-key-layer-scaling \
               --dataloader-type cyclic \
               --log-validation-ppl-to-tensorboard"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 20 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--load $CHECKPOINT_PATH \
            --data-impl mmap \
            --data-path $DATA_PATH \
            --split 100,0,0 \
            --distributed-backend nccl"
        #     --save $CHECKPOINT_SAVE_PATH \

export PYTHONPATH="$ROOT_DIR/..":"$ROOT_DIR/../megatron/accelerator/custom_ops/cuda_ops":$PYTHONPATH

command="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       $ROOT_DIR/../pretrain_aquila.py \
       $AQUILA_7B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS"
echo "执行命令: "$command""
eval $command
