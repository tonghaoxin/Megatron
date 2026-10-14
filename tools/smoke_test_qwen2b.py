# -*- coding: utf-8 -*-
"""HF transformers 单机 forward smoke test：验证「权重 + tokenizer」整条链路。

跑之前先跑 fix_qwen2b_bias.py 补零 bias，否则会因 qkv bias 形状不匹配而崩。

用法：
    SAVE_LOG_FILE_WITH_RANK_ID=false /opt/venv/xmlir/bin/python tools/smoke_test_qwen2b.py --model-dir /workspace/hf_qwen2b
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
    m = a.model_dir

    print("[1/4] 加载 tokenizer ...")
    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=False)
    print("      vocab_size:", tok.vocab_size)

    print("[2/4] 加载模型 (float32, CPU，最稳) ...")
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print("      参数量: %.3f B" % (n_params / 1e9))

    print("[3/4] forward pass ...")
    prompt = "The patient presents with"
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits
    print("      logits shape:", tuple(logits.shape), " (期望 (1, N, 151936))")
    print("      全部有限(finite):", bool(torch.isfinite(logits).all()))
    top = int(logits[0, -1].argmax())
    print("      argmax token id:", top, "->", repr(tok.decode([top])))

    print("[4/4] 贪心生成 16 个 token ...")
    gen = model.generate(**enc, max_new_tokens=16, do_sample=False)
    print("      生成:", repr(tok.decode(gen[0], skip_special_tokens=False)))
    print("=== SMOKE TEST OK ===")


if __name__ == "__main__":
    main()
