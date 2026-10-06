#! /bin/bash
#
# 昆仑芯 P800 四机 Llama2-70B 多机多卡推理（原地加载：不搬数据、不重切分、不改 Python 代码）
# ------------------------------------------------------------------
# 放置位置：/workspace/Megatron/examples/   （与训练脚本同级，才能 source 到 ../config.sh）
#
# 启动方式（每台机器上各执行一次，先起 NODE_RANK=0 那台）：
#   NODE_RANK=0 bash xpu_p800_run_llama2_70b_inference_4x4.sh
#   NODE_RANK=1 bash xpu_p800_run_llama2_70b_inference_4x4.sh
#   NODE_RANK=2 bash xpu_p800_run_llama2_70b_inference_4x4.sh
#   NODE_RANK=3 bash xpu_p800_run_llama2_70b_inference_4x4.sh
#
# 前置条件：
#   1) 训练进程已全部停止，四台 xpu-smi 确认显存已回落
#   2) 四台各自的 /workspace/checkpoints_4_4_2/iter_0000500 保持原样（不要搬、不要删）
#
# 并行配置：TP4 x PP4 x DP1 = world 16，即四台机器各出 4 张卡
#   —— 与训练时的切分刀法完全一致，所以磁盘上 16 片的形状就是代码期望的形状
#   —— 且每台机器本地正好存有自己需要的那 4 片，零搬迁
#   —— 必须 DP=1：run_llama_text_generation_server.py:67 起 HTTP 服务的条件是
#      is_pipeline_first_stage() and tp_rank==0，DP=2 会有两个进程抢同一个 5000 端口
# ------------------------------------------------------------------
set -e
set -x

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

#################################
# 0. 先把命令行传入的 NODE_RANK / MASTER_PORT 存下来
#    （config.sh 里 export 了 NODE_RANK=0，source 之后会覆盖命令行传进来的值，
#      所以必须先保存、source 完再覆盖回去）
#################################
IN_NODE_RANK=${NODE_RANK:-}
IN_MASTER_PORT=${MASTER_PORT:-}
IN_MASTER_ADDR=${MASTER_ADDR:-}
IN_CKPT_PATH=${CHECKPOINT_PATH:-}

#################################
# config（复用训练用的 config.sh）
#################################
source "$ROOT_DIR/../config.sh"

NODE_RANK=${IN_NODE_RANK:-${NODE_RANK:-0}}
MASTER_PORT=${IN_MASTER_PORT:-9917}          # 与训练的 9916 错开
MASTER_ADDR=${IN_MASTER_ADDR:-${MASTER_ADDR:-"192.168.0.15"}}
CHECKPOINT_PATH=${IN_CKPT_PATH:-${CHECKPOINT_PATH:-"/workspace/checkpoints_4_4_2"}}

export NODE_RANK
export MASTER_PORT
export MASTER_ADDR

#################################
# 基本配置
#################################
export XMLIR_LOG_PREFIX=infer_node${NODE_RANK}
export XMLIR_FC_AUTOTUNE_BOOL=false

TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/llama2_70b_data/tokenizer.model"}
LOAD_ITER=${LOAD_ITER:-500}                  # 要加载的迭代号

export LD_LIBRARY_PATH=${SO_PATH:-"$ROOT_DIR/../xccl-so:$ROOT_DIR/../xre-so:$LD_LIBRARY_PATH"}

# ---- 关键：每台只用 4 张卡 => world = 4 机 x 4 卡 = 16 = TP4 x PP4 x DP1 ----
export CUDA_VISIBLE_DEVICES="0,1,2,3"
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

NNODES=${NNODES:-4}
TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-4}

#################################
# 锁定到 iter 500：软链目录 + tracker 文件
# 目的：训练已跑到更后面的迭代时，latest_checkpointed_iteration.txt 会指向最后一版，
#       这里用一个独立目录把它钉死在 500
#################################
ITER_DIR="iter_$(printf '%07d' $LOAD_ITER)"
CKPT_LINK=${CKPT_LINK:-/workspace/ckpt_iter${LOAD_ITER}}
mkdir -p "$CKPT_LINK"
ln -sfn "$CHECKPOINT_PATH/$ITER_DIR" "$CKPT_LINK/$ITER_DIR"
printf '%d' $LOAD_ITER > "$CKPT_LINK/latest_checkpointed_iteration.txt"

#################################
# ★启动前自检：确认本节点需要的 4 个 model 分片都在
#
# 目录命名规则（megatron/checkpointing.py:96）：
#     mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}/model_rng.pt
#
# 本环境 world=16 / TP=4 / nproc_per_node=4，global_rank = NODE_RANK*4 + local：
#     tp_rank = global_rank % 4  = local
#     pp_rank = global_rank // 4 = NODE_RANK
# 所以第 N 台机器需要的是 mp_rank_{00..03}_{N:03d}，正好是它训练时自己存的那 4 片。
#################################
echo "=============================================================="
echo " 本节点 NODE_RANK=${NODE_RANK}  checkpoint 分片自检"
echo " 加载目录: ${CKPT_LINK}/${ITER_DIR}"
echo "=============================================================="

PP_RANK=$(printf '%03d' $NODE_RANK)
MISSING=0
for TP in $(seq 0 $((TP_SIZE-1))); do
    DIR_NAME=$(printf "mp_rank_%02d_%03d" $TP $NODE_RANK)
    F="$CKPT_LINK/$ITER_DIR/$DIR_NAME/model_rng.pt"
    if [ -f "$F" ]; then
        echo "  OK    $DIR_NAME/model_rng.pt  $(du -h "$F" | cut -f1)"
    else
        echo "  MISS  $F"
        MISSING=$((MISSING+1))
    fi
done

if [ "$MISSING" -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo " 本节点缺少 $MISSING 个 model 分片，禁止启动。"
    echo " 该目录下现有的 mp_rank_*:"
    ls -1 "$CKPT_LINK/$ITER_DIR" 2>/dev/null | head -40
    echo " 请确认 CHECKPOINT_PATH 是否指向本机的 /workspace/checkpoints_4_4_2"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi
echo " 自检通过：本节点 4 个分片齐全"

ulimit -c 0

#################################
# ★前置预检：设备节点 / 残留进程 / 显存
#    失败只打印警告，不中断（脚本开了 set -e，所有预检命令都带 || true）
#################################
echo "---- 预检 ----"
echo -n "  /dev 下的 XPU 设备节点数: "; ls /dev 2>/dev/null | grep -ciE "xpu|kl3|xdr" || true
echo -n "  /dev/xdrdrv: "; (ls -l /dev/xdrdrv 2>/dev/null && echo "存在") || echo "不存在（BKCL_ENABLE_XDR=1 会失效）"
echo -n "  /proc/kunlun: "; (ls /proc/kunlun >/dev/null 2>&1 && echo "存在") || echo "不存在（下面的超时设置会失败）"
_left=$(ps -ef 2>/dev/null | grep -E "torchrun|pretrain_llama|run_llama_text" | grep -v grep || true)
if [ -n "$_left" ]; then
    echo "  [!] 发现残留进程："
    echo "$_left" | sed 's/^/      /'
else
    echo "  OK：无残留 torchrun / 训练进程"
fi
echo "---- 预检结束 ----"

export NCCL_SOCKET_IFNAME=ens11np0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE
                  --nnodes $NNODES
                  --node_rank $NODE_RANK
                  --master_addr $MASTER_ADDR
                  --master_port $MASTER_PORT"

#################################
# 模型结构参数：必须与训练脚本逐项一致
#
# ★三项绝不能漏（漏了不会报 shape mismatch，而是静默加载失败或用错路径）：
#   1) --sequence-parallel        训练时开着，它改变 embedding/layernorm 的切分方式
#   2) --use-distributed-optimizer
#      决定了加载时去找哪个文件名：
#        True  -> mp_rank_{tp}_{pp}/model_rng.pt        （我们存的这个）
#        False -> mp_rank_{tp}_{pp}/model_optim_rng.pt  （不存在，会走到随机初始化）
#      （checkpointing.py:98-104；我们 --no-load-optim，所以不会去读那批 25G 的 optim 分片）
#   3) --use-cpu-initialization   避免在卡上初始化 70B 权重时爆显存（训练脚本同样开着）
#
# 注意：--out-seq-length / --temperature / --top_p / --top_k 在 server 模式下**完全不生效**，
#       它们只在 generate_samples.py 那种一次性脚本里用。server 模式下这些值由 HTTP 请求体决定
#       （见 megatron/text_generation_server.py:32-231，默认 tokens_to_generate=64）。
#################################
LLAMA2_70B_ARGS="--tensor-model-parallel-size ${TP_SIZE} \
                 --pipeline-model-parallel-size ${PP_SIZE} \
                 --num-layers 80 \
                 --hidden-size 8192 \
                 --ffn-hidden-size 28672 \
                 --num-attention-heads 64 \
                 --micro-batch-size 1 \
                 --seq-length 4096 \
                 --max-position-embeddings 4096 \
                 --tokenizer-type LLaMASentencePieceTokenizer \
                 --tokenizer-model $TOKENIZER_PATH \
                 --sequence-parallel \
                 --rmsnorm-epsilon 1e-6 \
                 --activation-func swiglu \
                 --use-rotary-position-embeddings \
                 --untie-embeddings-and-output-weights \
                 --attention-dropout 0 \
                 --hidden-dropout 0 \
                 --embedding-dropout 0 \
                 --disable-bias-linear \
                 --no-bias-gelu-fusion \
                 --no-position-embedding \
                 --no-masked-softmax-fusion \
                 --attention-softmax-in-fp32 \
                 --no-query-key-layer-scaling \
                 --multi-query-attention \
                 --multi-query-group-num 8 \
                 --ffn-dim-multiplier 1.3 \
                 --multiple-of 4096 \
                 --use-flash-attn \
                 --fused-rmsnorm \
                 --memory-saving \
                 --use-distributed-optimizer \
                 --use-cpu-initialization \
                 --apply-rotary-positional-embedding-kernel \
                 --seed 42 \
                 --fp16"

LOAD_ARGS="--load ${CKPT_LINK} \
           --no-load-optim \
           --no-load-rng \
           --exit-on-missing-checkpoint \
           --distributed-backend nccl"

#################################
# Runtime
#################################
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1

#################################
# XMLIR
#################################
export DIST_MULTI_STREAM=true
export XMLIR_DIST_SINGLETON_STREAM=true
export XMLIR_FA_GEMM_TYPE=float16
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true

#################################
# BKCL
#################################
export BKCL_CCIX_RING=1
export BKCL_CCIX_BUFFER_GM=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_BUFFER_GM=1
export BKCL_RDMA_PROXY_DISABLE=1
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=0
export XPU_ZEBU_MODE=1
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"ens11np0,ens12np0,ens13np0,ens14np0,ens15np0,ens16np0,ens17np0,ens18np0"}
export BKCL_XLINK_D2D=0
export BKCL_XLINK_C2C=1
export BKCL_XLINK_ETH=0
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=360000
unset BKCL_KL3_SYSCON_FLAG

#################################
# ★设备看门狗：与训练脚本逐行一致（训练脚本 189-204 行）
#
# 昆仑芯 XPU 自带硬件级 task timeout 看门狗，"whosyourdaddy" 是解锁隐藏接口的魔术串，
# 解锁后才能把单 task 判定阈值从默认值放宽到 3600000ms（1 小时）。
# 不设会怎样：推理首请求含 4096 长度 prefill + 首次图编译，单 task 可能远超默认阈值
#             → 触发 XPU 硬件超时 → 全卡连锁 Xid → 进程 abort（且没有 Python traceback）。
# 训练脚本同样开着 set -e 却能跑到 5000 步，说明本机 /proc/kunlun 确实存在且可写
# （否则这些 echo 早就让训练脚本在第 189 行退出了）。
#################################
if [ -d /proc/kunlun ]; then
    for i in 0 1 2 3 4 5 6 7; do
        echo "whosyourdaddy" >> /proc/kunlun/dev${i}/info
        echo "3600000" > /proc/kunlun/dev${i}/task_timeout_detect_threshold_in_ms
    done
    echo " 已放宽 XPU task timeout 到 3600000ms"
else
    echo " [!] /proc/kunlun 不存在，跳过超时设置（本次运行仅适用于短请求）"
fi

#################################
# 启动推理服务
# 只有 NODE_RANK=0 的 local_rank 0 会起 Flask，监听 0.0.0.0:5000
# 客户端连这台机器的 IP:5000
#################################
XFLAGS --enable megatron_aiak
PYTHONPATH="$ROOT_DIR/..":"$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops":$PYTHONPATH \
    torchrun $DISTRIBUTED_ARGS \
    $ROOT_DIR/../tools/run_llama_text_generation_server.py \
    $LLAMA2_70B_ARGS \
    $LOAD_ARGS
