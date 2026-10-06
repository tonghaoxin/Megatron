#!/bin/bash
###############################################################################
# 昆仑芯 P800 —— 推理环境「逐步 import」探针
#
# 目的：定位 2026-10-06 的哑巴式崩溃
#   现象：4 个 worker 打完 SYMBOL_REWRITE torch success 后 ~5 秒集体 exit 1，
#         全程没有任何 Python traceback。
#
# 与本脚本的区别（为什么它能看到 torchrun 看不到的东西）：
#   1) 单进程、不用 torchrun 包着 —— 不存在子进程被 SIGTERM 收走、输出被埋的情况；
#   2) python -u + PYTHONPATH/stdout 每次都显式 flush —— Python 3.8 在
#      「输出重定向到文件」时 stdout/stderr 是块缓冲的，进程被硬杀时缓冲区内容会丢；
#      本脚本每步先打印步骤名再执行，所以**哪怕进程被硬杀，也能知道死在哪一步**；
#   3) 每个 import 用 try/except 包起来并打印 traceback，一次跑完拿到全部信息。
#
# 用法（放 /workspace/Megatron/examples/ 下，任选一台：建议先 node0）
#     cd /workspace/Megatron/examples && bash xpu_import_probe.sh 2>&1 | tee /workspace/probe.log
# 或后台：
#     cd /workspace/Megatron/examples && nohup bash xpu_import_probe.sh > /workspace/probe.log 2>&1 &
#     tail -f /workspace/probe.log
#
# 预计耗时：30 秒 ~ 3 分钟（import torch 本身就要几十秒）。
###############################################################################

ROOT_DIR=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")

# 与训练完全同一套环境变量
source "$ROOT_DIR/../config.sh"

export LD_LIBRARY_PATH=$SO_PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTHONUNBUFFERED=1              # 关键：防止缓冲区丢内容
export PYTHONDONTWRITEBYTECODE=1

echo "=============================================================="
echo " XPU 推理环境逐步 import 探针"
echo " HOST            : $(hostname)"
echo " CUDA_VISIBLE    : $CUDA_VISIBLE_DEVICES"
echo " LD_LIBRARY_PATH : $LD_LIBRARY_PATH"
echo " PYTHON          : $(command -v python || echo /opt/venv/xmlir/bin/python)"
echo "=============================================================="
echo " 注意：每一步都是「先打印步骤名(fLUSH) -> 再执行」，"
echo "       所以即使某一步让进程被硬杀，最后看到的那个步骤名就是死亡点。"
echo "=============================================================="

# XFLAGS 会把 megatron_aiak 相关的符号改写打开（与推理脚本保持一致）
XFLAGS --enable megatron_aiak 2>&1 | tail -5

/opt/venv/xmlir/bin/python -u - <<'PYEOF'
import os, sys, time, traceback

def step(n, name, fn):
    # 先写步骤名并 flush：万一这一步把进程干掉，至少知道死在哪
    sys.stdout.write("  [%02d] %-46s " % (n, name))
    sys.stdout.flush()
    t0 = time.time()
    try:
        ret = fn()
        sys.stdout.write("OK   %.1fs  %s\n" % (time.time() - t0, ret if ret is not None else ""))
    except BaseException as e:
        sys.stdout.write("FAIL %.1fs  %s: %s\n" % (time.time() - t0, type(e).__name__, e))
        sys.stdout.flush()
        traceback.print_exc()
    sys.stdout.flush()

def imp(name):
    def f():
        import importlib
        m = importlib.import_module(name)
        return getattr(m, "__file__", "(no __file__)")
    return f

# 推理入口脚本自己会 sys.path.append(dirname(__file__)/..)，也就是 /workspace/Megatron。
# 探针是单进程跑的，所以这里手动补一模一样的路径，否则会误报 ModuleNotFoundError。
MEGATRON_ROOT = os.environ.get("PROBE_MEGATRON_ROOT", "/workspace/Megatron")
print("=== START probe  pid=%d  cwd=%s  megatron_root=%s ===" % (os.getpid(), os.getcwd(), MEGATRON_ROOT), flush=True)

step(0,  "sys.path 补上 Megatron 根（等价入口脚本第 7-8 行）", lambda: sys.path.insert(0, MEGATRON_ROOT) or MEGATRON_ROOT)

step(1,  "python / ld flags", lambda: "py%s" % sys.version.split()[0])
step(2,  "import torch", imp("torch"))
step(3,  "import torch.distributed", imp("torch.distributed"))

def dev():
    import torch
    return "device_count=%d  is_available=%s" % (torch.cuda.device_count(), torch.cuda.is_available())
step(4,  "torch.cuda device_count / available", dev)

def matmul():
    import torch
    a = torch.randn(256, 256, device="cuda:0")
    c = (a @ a).sum().item()
    return "sum=%.3f  dev=%s" % (c, a.device)
step(5,  "XPU 上跑一个小 matmul（验证卡真能用）", matmul)

step(6,  "import numpy", imp("numpy"))
step(7,  "import flask（HTTP 服务依赖）", imp("flask"))
step(8,  "import sentencepiece（tokenizer 依赖）", imp("sentencepiece"))
step(9,  "import megatron", imp("megatron"))
step(10, "import megatron.accelerator", imp("megatron.accelerator"))

def acc():
    from megatron.accelerator import get_accelerator
    a = get_accelerator()
    return "backend=%s  available=%s  device=%s" % (a.backend_name, a.is_available(), a.device_name())
step(11, "get_accelerator() 可用性检查（12 行的 assert 同款）", acc)

step(12, "import megatron.arguments", imp("megatron.arguments"))
step(13, "import megatron.initialize", imp("megatron.initialize"))
step(14, "import megatron.model (LLaMAModel)", imp("megatron.model"))
step(15, "import megatron.checkpointing", imp("megatron.checkpointing"))
step(16, "import megatron.text_generation", imp("megatron.text_generation"))
step(17, "import megatron.text_generation_server", imp("megatron.text_generation_server"))

def load_entry():
    # 把推理入口脚本当源码编译一遍，验证它自身语法 OK（不执行它）
    p = "/workspace/Megatron/tools/run_llama_text_generation_server.py"
    src = open(p, "rb").read()
    compile(src, p, "exec")
    return "syntax OK, %d bytes" % len(src)
step(18, "编译 run_llama_text_generation_server.py（语法检查）", load_entry)

print("=== ALL DONE ===  （任何一步 FAIL 或进程提前消失，都以上面最后一个步骤名为准）", flush=True)
PYEOF

echo "=============================================================="
echo " 探针结束。退出码: $?"
echo "=============================================================="
