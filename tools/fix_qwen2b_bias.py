# -*- coding: utf-8 -*-
"""给 bias-free 的 Qwen2 权重补上 q/k/v 的零 bias，使其符合标准 Qwen2 形状（HF/vLLM 可直接加载）。

背景：
- 训练用了 --disable-bias-linear，故 qkv 无 bias（219 个 tensor 全为 weight）。
- 但 HF 的 Qwen2 实现（4.41/4.44 实测）把 q/k/v 的 bias 硬编码为 True，不读 config.qkv_bias。
- 因此必须给 q/k/v 补 24*3=72 个全零 bias（y=Wx+0 与 y=Wx 数学等价，不改变结果），
  同时把 config.json 的 qkv_bias 改为 true、torch_dtype 修正为 float16。

用法：
    SAVE_LOG_FILE_WITH_RANK_ID=false PYTHONPATH= /opt/venv/xmlir/bin/python tools/fix_qwen2b_bias.py --model-dir /workspace/hf_qwen2b
"""
import argparse
import json
import os

os.environ.setdefault("SAVE_LOG_FILE_WITH_RANK_ID", "false")

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/workspace/hf_qwen2b")
    a = ap.parse_args()

    bin_path = os.path.join(a.model_dir, "pytorch_model.bin")
    cfg_path = os.path.join(a.model_dir, "config.json")

    # ---- 1. 修 config.json ----
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["qkv_bias"] = True            # 标准 Qwen2：qkv 带 bias（下面补零，等价于无 bias）
    cfg["mlp_bias"] = False           # MLP 无 bias，保持显式 false
    cfg["torch_dtype"] = "float16"    # 修正非法值 "fp16"
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("[config] qkv_bias=%s mlp_bias=%s torch_dtype=%s"
          % (cfg["qkv_bias"], cfg.get("mlp_bias"), cfg.get("torch_dtype")))

    # ---- 2. 补零 bias ----
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", hidden // heads)
    n_layers = cfg["num_hidden_layers"]
    dtype = torch.float16

    sd = torch.load(bin_path, map_location="cpu")
    added = 0
    for i in range(n_layers):
        base = "model.layers.%d.self_attn." % i
        for name, size in [("q_proj.bias", heads * head_dim),
                           ("k_proj.bias", kv * head_dim),
                           ("v_proj.bias", kv * head_dim)]:
            k = base + name
            if k not in sd:
                sd[k] = torch.zeros(size, dtype=dtype)
                added += 1
    print("[weights] 新增零 bias tensor 数: %d (期望 72)" % added)
    torch.save(sd, bin_path)
    print("[weights] 已写回: %s" % bin_path)
    print("[done] 总 tensor 数应为 219 + 72 = %d，实际 %d" % (219 + 72, len(sd)))


if __name__ == "__main__":
    main()
