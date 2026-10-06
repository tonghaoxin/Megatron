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
# 0. 先把命令行传入的可能被 config.sh 覆盖的变量存下来
#    （config.sh 里 export 了 NODE_RANK=0 / NNODES=4 / MASTER_ADDR / MASTER_PORT /
#      CHECKPOINT_PATH，source 之后会覆盖命令行传进来的值，所以必须先保存、source 完再还原）
#
#    ★这里踩过两次坑：
#      第一次：只保护了 NODE_RANK，四台启动时 NODE_RANK 全变成 0
#      第二次：NNODES 没保护，单机自检传 NNODES=1 被 config.sh 的 4 吃掉，
#              torchrun 仍在等四节点 rendezvous → 卡住不动
#    教训：config.sh 里出现过的每一个变量，这里都要保护。
#################################
IN_NODE_RANK=${NODE_RANK:-}
IN_NNODES=${NNODES:-}
IN_MASTER_PORT=${MASTER_PORT:-}
IN_MASTER_ADDR=${MASTER_ADDR:-}
IN_CKPT_PATH=${CHECKPOINT_PATH:-}
IN_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

#################################
# config（复用训练用的 config.sh）
#################################
source "$ROOT_DIR/../config.sh"

NODE_RANK=${IN_NODE_RANK:-${NODE_RANK:-0}}
NNODES=${IN_NNODES:-${NNODES:-4}}
MASTER_PORT=${IN_MASTER_PORT:-9917}          # 与训练的 9916 错开
MASTER_ADDR=${IN_MASTER_ADDR:-${MASTER_ADDR:-"192.168.0.15"}}
CHECKPOINT_PATH=${IN_CKPT_PATH:-${CHECKPOINT_PATH:-"/workspace/checkpoints_4_4_2"}}

export NODE_RANK
export NNODES
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
export CUDA_VISIBLE_DEVICES=${IN_CUDA_VISIBLE_DEVICES:-"0,1,2,3"}
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`

TP_SIZE=${TP_SIZE:-4}
PP_SIZE=${PP_SIZE:-4}

#################################
# ★启动参数回显：每次都打一行，防止再出现"传了 NNODES=1 却没生效"这种哑亏
#################################
echo "=============================================================="
echo " 启动配置: NODE_RANK=${NODE_RANK}  NNODES=${NNODES}  GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "           world = ${NNODES} x ${GPUS_PER_NODE} = $((NNODES * GPUS_PER_NODE))   (TP x PP = $((TP_SIZE * PP_SIZE)))"
echo "           MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "=============================================================="
if [ $((NNODES * GPUS_PER_NODE)) -ne $((TP_SIZE * PP_SIZE)) ]; then
    echo " [!] 注意：world($((NNODES * GPUS_PER_NODE))) != TP($TP_SIZE) x PP($PP_SIZE)，"
    echo "     会先撞 parallel_state.py:132 的 RuntimeError —— 单机自检时这是预期结果。"
fi

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
#   1) --sequence-parallel  ——★推理时必须关★（本脚本默认已关，见下方 SEQ_PARALLEL_ARG）
#      训练时开着，它改变 **激活向量**（embedding 输出 / LayerNorm / Linear 残差）的切分方式：
#      沿 **序列维 s** 切成 TP=4 份。训练时 s=4096，4096%4=0，正常。
#      但在线推理是 **逐 token 解码**：prefix 之后每次 forward 的 s 都 = 1，
#      1 % 4 != 0 ⇒ 直接撞 mappings.py:58
#           assert dim_size % world_size == 0,
#           "First dimension of the tensor should be divisible by tensor parallel size"
#      ★关掉它不影响 checkpoint 加载：sequence-parallel 只切 **激活**，不切 **权重**，
#        80 层 attention/mlp layernorm/Linear 的 shape 完全不受它影响，
#        16 片 .pt 的形状不变 ⇒ 原地加载照常成功。
#      想强行恢复（不推荐）：INF_SEQ_PARALLEL=1 bash 本脚本
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

#################################
# ★序列并行开关：推理默认 **关闭**
#   原因见上面第 1) 条。默认不传 --sequence-parallel，命令行上空一个参数位。
#################################
SEQ_PARALLEL_ARG=""
if [ "${INF_SEQ_PARALLEL:-0}" = "1" ]; then
    SEQ_PARALLEL_ARG="--sequence-parallel"
fi
echo " [i] sequence-parallel = ${SEQ_PARALLEL_ARG:-OFF （推理默认关闭，避免逐 token 解码时 s=1 无法被 TP 整除）}"

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
                 ${SEQ_PARALLEL_ARG} \
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
# ★PYTHONPATH：默认什么都不注入（这是 2026-10-06 单机定位出来的元凶候选）
#
# 为什么默认空：
#   1) 训练脚本那一行写的是**单引号**
#        PYTHONPATH='$ROOT_DIR/..':'$ROOT_DIR/../megatron/accelerator/custom_ops/xpu_ops':$PYTHONPATH
#      变量不展开 ⇒ 实际注入的是两个带美元号的无效路径 ⇒ **训练时这两个路径根本没生效**，
#      训练靠 pretrain_llama.py 所在目录自动进 sys.path[0] 才 import 到 megatron。
#   2) run_llama_text_generation_server.py 第 7-8 行自己做了
#        sys.path.append(dirname(__file__)/..)   == /workspace/Megatron
#      所以推理脚本也**不需要**外部注入 Megatron 根目录。
#   3) 2026-10-06 复现实测：注入 → 崩；**不注入 → 照样崩**，现象完全一致
#      （nnodes=1 / world=4 单机也一样），所以 PYTHONPATH **已被证伪**，不是元凶。
#      保留这个开关只为对齐训练脚本的实际行为、便于回退对比。
#
# 想恢复注入时：INF_EXTRA_PYTHONPATH="/workspace/Megatron:..." bash 本脚本
#################################
if [ -n "${INF_EXTRA_PYTHONPATH:-}" ]; then
    export PYTHONPATH="${INF_EXTRA_PYTHONPATH}:${PYTHONPATH}"
    echo " [i] 额外注入 PYTHONPATH: ${INF_EXTRA_PYTHONPATH}"
else
    echo " [i] 不注入任何 PYTHONPATH（与训练脚本的实际行为一致）"
fi

#################################
# ★输出缓冲：必须关掉
#
# Python 3.8 在「stdout/stderr 被重定向到文件」时是**块缓冲**（8KB 一刷），
# 进程若被硬杀 / 被 C 层 exit，缓冲区里的内容（包括 traceback）会全部丢失。
# 这就是为什么之前四次尝试都是「全程零 Python 输出 + exit 1」：
# 看得见的行（XCCL loaded / BKCL WARN / SYMBOL_REWRITE）都是 **C 层**直接写 stderr，
# 不受 Python 缓冲影响；而 Python 层写的东西可能压根没刷出来。
# 加上 PYTHONUNBUFFERED=1 之后，任何 print 都会立刻落盘，真正的报错才会现形。
#################################
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
echo " [i] PYTHONUNBUFFERED=1（关闭输出缓冲，避免报错被吞）"

#################################
# ★★日志去向（2026-10-06 排障的头号发现）
#
# `megatron/accelerator/xpu_accelerator.py` 顶层（不在函数里，import 就执行）：
#       rank      = os.getenv("RANK", "0")
#       dir_path  = "./logs" + strftime("%Y-%m-%d_%H_") + str(getenv("XMLIR_LOG_PREFIX"))
#       log_file  = dir_path + "/" + rank + ".log"
#       if getenv("SAVE_LOG_FILE_WITH_RANK_ID", "true") == "true":
#           fd = os.open(log_file, O_WRONLY|O_CREAT|O_TRUNC)
#           os.dup2(fd, 1);  os.dup2(fd, 2)      # ← stdout/stderr 被整个换掉
#
# 而触发时机极早：megatron/__init__.py 会 `from .initialize import initialize_megatron`，
# initialize.py 第 27 行在 **模块级** 就跑了 `get_accelerator().get_operator(...)`，
# 于是 get_accelerator() → import xpu_accelerator → dup2。
# ⇒ **只要 import megatron 成功，之后所有 print / traceback 全进那个 ./logs... 目录**，
#    nohup 重定向的那个文件从此变哑（只剩之前 C 层已经写出去的 XCCL/BKCL 那些行）。
#    这就是为什么四次尝试都是「SYMBOL_REWRITE 之后一片空白 + exit 1」。
#    （顺带解释：训练时找到的 0.log / 31.log 就是这个机制产生的。）
#
# 默认关掉：设 false 后一切回到本脚本的日志里，排障最省事。
# 想恢复昆仑芯的分 rank 日志：INF_KEEP_RANK_LOG=1 bash 本脚本
#################################
if [ "${INF_KEEP_RANK_LOG:-0}" = "1" ]; then
    export SAVE_LOG_FILE_WITH_RANK_ID=true
    echo " [i] 保留昆仑芯分 rank 日志 -> ./logs<日期>_${XMLIR_LOG_PREFIX:-None}/<RANK>.log"
else
    export SAVE_LOG_FILE_WITH_RANK_ID=false
    echo " [i] SAVE_LOG_FILE_WITH_RANK_ID=false —— 所有输出回到本日志文件（排障用）"
fi

#################################
# 启动推理服务
# 只有 NODE_RANK=0 的 local_rank 0 会起 Flask，监听 0.0.0.0:5000
# 客户端连这台机器的 IP:5000
#################################
XFLAGS --enable megatron_aiak

if [ "${INF_DIRECT:-0}" = "1" ]; then
    ###########################################################################
    # 排障模式：不用 torchrun，单进程直接跑 python -u
    #
    # 为什么要这一步：torchrun 会把 4 个子进程的输出混在同一个文件里，而且
    # 它自己会先把 XCCL/BKCL/SYMBOL_REWRITE 那一堆 C 层日志打出来，夹在中间，
    # 让"到底有没有 Python 报错"变得很难读。这里把 torchrun 整个摘掉：
    #   - 单进程、python -u（无缓冲）→ 任何 traceback 都会原样落盘
    #   - world=1 必然撞 parallel_state 的 RuntimeError，
    #     看到那行 = 前面所有 import / 参数解析都过了（"到此一游"标记）
    # 用法： INF_DIRECT=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=9918 bash 本脚本
    ###########################################################################
    echo " [!] INF_DIRECT=1 —— 单进程直跑模式（排障用，不会起服务）"
    export RANK=0
    export WORLD_SIZE=1
    export LOCAL_RANK=0
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-9918}
    cd "$ROOT_DIR/.." || exit 1
    /opt/venv/xmlir/bin/python -u tools/run_llama_text_generation_server.py \
        $LLAMA2_70B_ARGS \
        $LOAD_ARGS
    echo "---- 直跑结束，退出码: $? ----"
else
    torchrun $DISTRIBUTED_ARGS \
        $ROOT_DIR/../tools/run_llama_text_generation_server.py \
        $LLAMA2_70B_ARGS \
        $LOAD_ARGS
fi
