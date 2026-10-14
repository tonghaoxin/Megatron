# -*- coding: utf-8 -*-
"""XPU 版 smoke test：在训练容器 xpytorch 里把自训 Qwen2 权重搬到 XPU 上跑 forward + generate。

定位：这是「CPU 冒烟 → XPU 纯 PyTorch → vllm_kunlun」三层递进的第二层。
只验证 XPU 基础算子能不能正确吃下这份权重，**不代表 vllm**（vllm 有自己另一套算子库）。

与 CPU 版 smoke_test_qwen2b.py 的差异：
  1. torch_dtype=float16（贴近 vllm 实际）
  2. model / 输入 显式 .to("cuda")（昆仑芯训练容器把 XPU 伪装成 cuda，无 torch.xpu 接口）
  3. attn_implementation="eager"（绕开 sdpa 在 xpytorch 上可能缺失的 XPU 后端）

用法：
    SAVE_LOG_FILE_WITH_RANK_ID=false /opt/venv/xmlir/bin/python tools/smoke_test_qwen2b_xpu.py --model-dir /workspace/hf_qwen2b
"""
import argparse
import os

os.environ.setdefault("SAVE_LOG_FILE_WITH_RANK_ID", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/hf_qwen2b")
    a = ap.parse_args()

    print("[check] torch:", torch.__version__,
          "| cuda is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            dev_name = torch.cuda.get_device_name(0)
        except Exception:
            dev_name = "?"
        print("[check] cuda device_count:", torch.cuda.device_count(),
              "| device_name:", dev_name)

    print("[1/4] 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(a.model_dir)

    print("[2/4] 加载模型 (fp16) 并搬到 XPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, torch_dtype=torch.float16, attn_implementation="eager")
    model = model.to("xpu")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print("      参数量: %.3f B" % (n_params / 1e9))

    print("[3/4] XPU 上 forward pass ...")
    prompt = "The patient presents with"
    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to("xpu")
    attention_mask = enc["attention_mask"].to("xpu")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits
    print("      logits shape:", tuple(logits.shape), "| dtype:", logits.dtype)
    print("      全部有限(finite):", bool(torch.isfinite(logits).all()))
    top = int(logits[0, -1].argmax().item())
    print("      argmax token id:", top, "->", repr(tok.decode([top])))

    print("[4/4] XPU 上贪心生成 8 个 token ...")
    gen = model.generate(input_ids, attention_mask=attention_mask,
                         max_new_tokens=8, do_sample=False)
    print("      生成:", repr(tok.decode(gen[0].cpu(), skip_special_tokens=False)))
    print("=== XPU SMOKE TEST OK ===")


if __name__ == "__main__":
    main()
