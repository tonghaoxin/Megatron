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
DATA_PATH=${TRAIN_DATA_DIR}/datasets/megatron_llama/pile_llama_test/pile-llama_text_document
CHECKPOINT_PATH=${TRAIN_DATA_DIR}/megatron_checkpoint/megatron_llama/megatron_llama_7b_checkpoint_tp1_pp1_dp8_zero1
TOKENIZER_PATH=${TRAIN_DATA_DIR}/tokenizers/megatron_llama/llama_tokenizer/tokenizer.model

TENSORBOARD_PATH=./tensorborads/llama-7b

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-9999}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}

TP_SIZE=1
PP_SIZE=1

############################################ 模型训练参数 ############################################

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

LLAMA_7B_ARGS="--tensor-model-parallel-size ${TP_SIZE} \
               --pipeline-model-parallel-size ${PP_SIZE} \
               --num-layers 32 \
               --hidden-size 4096 \
               --ffn-hidden-size 11008 \
               --num-attention-heads 32 \
               --micro-batch-size 4 \
               --global-batch-size 128 \
               --seq-length 1024 \
               --max-position-embeddings 1024 \
               --lr 1.0e-5 \
               --min-lr 1.0e-5 \
               --clip-grad 1.0 \
               --optimizer adam \
               --adam-beta1 0.9 \
               --adam-beta2 0.95 \
               --weight-decay 0.1 \
               --adam-eps 1e-08 \
               --train-iters 200 \
               --lr-decay-iters 320000 \
               --lr-decay-style cosine \
               --lr-warmup-fraction .01 \
               --no-async-tensor-model-parallel-allreduce \
               --fp16
               --tokenizer-type LLaMASentencePieceTokenizer \
               --tokenizer-model ${TOKENIZER_PATH} \
               --swiglu \
               --use-rotary-position-embeddings \
               --rmsnorm-epsilon 1e-6 \
               --fused-rmsnorm \
               --no-position-embedding \
               --disable-bias-linear \
               --attention-dropout 0 \
               --hidden-dropout 0 \
               --embedding-dropout 0 \
               --use-distributed-optimizer \
               --use-flash-attn \
               --untie-embeddings-and-output-weights"
            #    --apply-rotary-positional-embedding-kernel \
            #    --tp-gemm-nccl-parallel \

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 10000 \
             --eval-interval 1000 \
             --eval-iters 10"

OTHER_ARGS="--load $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --data-impl mmap \
            --split 949,50,1 \
            --distributed-backend nccl"
       #      --save $CHECKPOINT_PATH \

export PYTHONPATH="$ROOT_DIR/..":"$ROOT_DIR/../megatron/accelerator/custom_ops/cuda_ops":$PYTHONPATH

command="python -m torch.distributed.launch $DISTRIBUTED_ARGS \
      $ROOT_DIR/../pretrain_llama.py \
      $LLAMA_7B_ARGS \
      $OUTPUT_ARGS \
      $OTHER_ARGS"
echo "执行命令: "$command""
eval $command
