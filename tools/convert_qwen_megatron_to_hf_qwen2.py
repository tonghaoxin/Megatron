#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把昆仑芯 fork 的 Megatron-LM Qwen checkpoint 转换成 HuggingFace **Qwen2** 标准格式。

适用范围（重要）
----------------
仅支持 **TP=1 且 PP=1** 的 checkpoint（也就是单机 DP 训练出来的那种）。
理由是：TP>1 时 query_key_value / dense_h_to_4h 是按列切开的，而 GQA 的
qkv 是「按 query group 交错」排列，跨 TP 合并要先把每个分片再按 group 拆开重排，
逻辑复杂且极易出错。若你的 ckpt 是 TP>1，请先用 tools/checkpoint_util.py 重切分到 TP=1。

产出目录结构与从 HF Hub 下载的 Qwen2 模型一致：
    config.json
    pytorch_model.bin（或 model.safetensors）
    tokenizer 相关文件（若给了 --tokenizer-path）

用法
----
# 1) 先探查结构，确认 key 名对得上（不写文件）
python tools/convert_qwen_megatron_to_hf_qwen2.py \
    --load-path /workspace/ckpt_qwen2b/iter_0000500 --print-structure

# 2) 正式转换
python tools/convert_qwen_megatron_to_hf_qwen2.py \
    --load-path /workspace/ckpt_qwen2b/iter_0000500 \
    --save-path /workspace/hf_qwen2b \
    --tokenizer-path /workspace/models

三个必须核对的数字（弄错不会报错，只会让模型变傻）
--------------------------------------------------
1. --rope-theta 默认 **10000**。昆仑芯 fork 的 --rotary-emb-base 默认就是 10000。
   注意 Qwen2 官方 config 是 1000000，但那是人家训练时用的值；我们训练时没改过
   这个参数，所以这里必须写 10000，写成 1000000 会让位置编码整体错位。
2. config 的 **qkv_bias = false**。我们的训练脚本带了 --disable-bias-linear，
   权重里根本没有 qkv 的 bias。若这里写成 true，HF/vLLM 会「随机初始化」一份 bias，
   属于最难查的静默错误。
3. SwiGLU 的 gate/up 顺序。Megatron 的 activation_funcs.SwiGLU 是
   `silu(前半) * 后半`，即 dense_h_to_4h 的**前半是 gate、后半是 up**。
   本脚本按此拆分，已与源码核对（megatron/model/activation_funcs.py:70-71）。
"""

import os

os.environ.setdefault("SAVE_LOG_FILE_WITH_RANK_ID", "false")

import argparse
import json
import shutil
import sys

import torch


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Megatron(Qwen) -> HuggingFace Qwen2 checkpoint converter"
    )
    p.add_argument("--load-path", required=True,
                   help="Megatron checkpoint 目录，例如 .../ckpt_qwen2b/iter_0000500")
    p.add_argument("--save-path", default=None,
                   help="输出目录。给了 --print-structure 时可省略")
    p.add_argument("--tokenizer-path", default=None,
                   help="含 qwen.tiktoken 或 HF tokenizer 文件的目录，会被拷到输出目录")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                   help="输出权重 dtype，默认 fp16（与训练一致）")
    p.add_argument("--rope-theta", type=float, default=10000.0,
                   help="必须与训练时 --rotary-emb-base 一致。该 fork 默认 10000")
    p.add_argument("--max-position-embeddings", type=int, default=4096,
                   help="与训练时 --max-position-embeddings 一致")
    p.add_argument("--rms-norm-eps", type=float, default=1e-6,
                   help="与训练时 --rmsnorm-epsilon 一致")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="写进 config 的词表大小。默认=embedding 行数（即 padded 后的值）")
    p.add_argument("--eos-token-id", type=int, default=151643)
    p.add_argument("--print-structure", action="store_true",
                   help="只打印 Megatron checkpoint 的结构，不转换")
    p.add_argument("--save-safetensors", action="store_true",
                   help="尝试用 safetensors 保存（没装该库则自动退回 .bin）")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def find_model_file(load_path):
    """在 Megatron ckpt 目录里找到存权重的那个 .pt。

    TP=1/PP=1 + 分布式优化器时，目录结构通常是：
        mp_rank_00/model_rng.pt          <-- 权重（我们要的）
        mp_rank_00_000/optim.pt ...      <-- 优化器分片（不含 'rng'，会被跳过）
    """
    if load_path.endswith("/"):
        load_path = load_path[:-1]
    if not os.path.isdir(load_path):
        raise FileNotFoundError("load-path 不是目录: %s" % load_path)

    candidates = []
    for name in sorted(os.listdir(load_path)):
        d = os.path.join(load_path, name)
        if not os.path.isdir(d) or not name.startswith("mp_rank"):
            continue
        for f in sorted(os.listdir(d)):
            if "rng" in f and f.endswith(".pt"):
                candidates.append(os.path.join(d, f))
    if not candidates:
        raise FileNotFoundError(
            "在 %s 下没找到含 'rng' 的 .pt 文件。请确认路径指向的是 "
            "iter_XXXXXXX 这一层目录。" % load_path
        )
    # 目录名越短越可能是权重（mp_rank_00 优先于 mp_rank_00_000）
    candidates.sort(key=lambda x: (len(os.path.basename(os.path.dirname(x))), x))
    return candidates[0]


def flatten_state_dict(d, prefix=""):
    """把嵌套的 state_dict 拍平成 'a.b.c' 形式的平铺字典。

    该 fork 的 ckpt 嵌套层数不固定（本例实测是 model / language_model / encoder
    三层嵌套，再往下才是平铺的 dotted key），所以统一拍平后按后缀查找最稳，
    不用去猜每一层的字典名。
    """
    flat = {}
    if not isinstance(d, dict):
        return flat
    for k, v in d.items():
        key = ("%s.%s" % (prefix, k)) if prefix else str(k)
        if isinstance(v, dict):
            sub = flatten_state_dict(v, key)
            if sub:
                flat.update(sub)
            else:
                flat[key] = v
        else:
            flat[key] = v
    return flat


def get_path(sd, candidates):
    """在 ckpt 里按候选 key 列表找第一个存在的 tensor。

    candidates 里写「尽量完整」的路径；先精确匹配，再按后缀匹配（取最短命中）。
    """
    if isinstance(candidates, str):
        candidates = [candidates]
    flat = flatten_state_dict(sd, "")
    for c in candidates:
        if c in flat:
            return flat[c]
    for c in candidates:
        hits = [k for k in flat if k == c or k.endswith("." + c)]
        if hits:
            hits.sort(key=len)
            return flat[hits[0]]
    raise KeyError(
        "checkpoint 里找不到下列任何一个 key：\n  %s\n"
        "ckpt 中实际存在的 key（最多列 30 个）：\n  %s"
        % ("\n  ".join(candidates), "\n  ".join(sorted(flat.keys())[:30]))
    )


def layer_key(i, name):
    """第 i 层的某个参数，给出候选路径（兼容嵌套层数不同的 ckpt）。"""
    short = "layers.%d.%s" % (i, name)
    return ["model.language_model.encoder." + short,
            "language_model.encoder." + short,
            short]


# ---------------------------------------------------------------------------
# 转换
# ---------------------------------------------------------------------------
def split_qkv_gqa(qkv, num_heads, num_query_groups, head_dim):
    """拆 Megatron 的 fused qkv。

    Megatron 的排列是「按 query group 交错」（见 megatron/model/transformer.py:682-690）:

        [ g0_q( repeats*hd ), g0_k( hd ), g0_v( hd ),
          g1_q( repeats*hd ), g1_k( hd ), g1_v( hd ), ... ]

    其中 repeats = num_heads / num_query_groups。
    MHA（num_query_groups == num_heads）时 repeats=1，公式自动退化成
    [q,k,v] 三等分，同样正确。
    """
    out_dim, hidden = qkv.shape[0], qkv.shape[1]
    repeats = num_heads // num_query_groups
    group_dim = (repeats + 2) * head_dim
    if out_dim != num_query_groups * group_dim:
        raise ValueError(
            "qkv 形状对不上：out_dim=%d，但 num_query_groups*(repeats+2)*head_dim="
            "%d*(%d+2)*%d=%d。请核对 num_query_groups / head_dim。"
            % (out_dim, num_query_groups, repeats, head_dim,
               num_query_groups * group_dim)
        )
    v = qkv.view(num_query_groups, group_dim, hidden)
    q = v[:, : repeats * head_dim, :].reshape(num_heads * head_dim, hidden)
    k = v[:, repeats * head_dim : repeats * head_dim + head_dim, :].reshape(
        num_query_groups * head_dim, hidden
    )
    vv = v[:, repeats * head_dim + head_dim :, :].reshape(
        num_query_groups * head_dim, hidden
    )
    return q, k, vv


def convert(args):
    ckpt_file = find_model_file(args.load_path)
    print("[i] 加载: %s" % ckpt_file)
    sd = torch.load(ckpt_file, map_location="cpu")

    if args.print_structure:
        print("\n=== Megatron checkpoint 结构 ===")
        for k, v in sd.get("model", sd).items() if isinstance(sd.get("model", sd), dict) else []:
            print("  %s : %s" % (k, type(v).__name__))
        flat = flatten_state_dict(sd, "")
        print("\n--- 顶层部件（embedding / output_layer / final norm）---")
        for c in ["model.language_model.embedding.word_embeddings.weight",
                  "model.language_model.output_layer.weight",
                  "model.language_model.encoder.final_layernorm.weight"]:
            hits = [k for k in flat if k == c or k.endswith("." + c)]
            if not hits:
                hits = [k for k in flat if (c.split(".")[-2] + "." + c.split(".")[-1]) in k]
            if hits:
                hits.sort(key=len)
                val = flat[hits[0]]
                print("  %-60s %s" % (hits[0], tuple(val.shape) if hasattr(val, "shape") else val))
            else:
                print("  %-60s <未找到>" % c)

        keys0 = [k for k in flat if k.endswith(".layers.0." + k.split(".layers.0.")[-1])
                 and ".layers.0." in k]
        print("\n--- 第 0 层的 key 与形状 ---")
        for k in sorted(keys0):
            val = flat[k]
            print("  %-60s %s" % (k, tuple(val.shape) if hasattr(val, "shape") else val))
        margs = sd.get("args", None)
        if margs is not None:
            print("\n--- ckpt 里记录的训练参数（节选）---")
            for k in ["num_layers", "hidden_size", "num_attention_heads",
                      "num_query_groups", "ffn_hidden_size", "kv_channels",
                      "padded_vocab_size", "tensor_model_parallel_size",
                      "pipeline_model_parallel_size", "max_position_embeddings",
                      "params_dtype"]:
                if hasattr(margs, k):
                    print("  %-32s %s" % (k, getattr(margs, k)))
        return

    margs = sd.get("args", None)
    if margs is None:
        raise ValueError("checkpoint 里没有 args，无法自动推断模型结构。")

    # ---- 从 ckpt 的 args 读模型结构（最可靠，不要手填）----
    num_layers = int(margs.num_layers)
    hidden = int(margs.hidden_size)
    heads = int(margs.num_attention_heads)
    qg = int(getattr(margs, "num_query_groups", 1) or 1)
    ffn = int(margs.ffn_hidden_size)
    tp = int(getattr(margs, "tensor_model_parallel_size", 1))
    pp = int(getattr(margs, "pipeline_model_parallel_size", 1))
    head_dim = int(getattr(margs, "kv_channels", None) or (hidden // heads))

    print("[i] 从 ckpt args 读到的结构：")
    print("    layers=%d hidden=%d heads=%d query_groups=%d ffn=%d head_dim=%d"
          % (num_layers, hidden, heads, qg, ffn, head_dim))
    print("    TP=%d PP=%d" % (tp, pp))

    # ---- 硬约束：只支持 TP=1 PP=1 ----
    if tp != 1 or pp != 1:
        raise ValueError(
            "本脚本只支持 TP=1 且 PP=1 的 checkpoint，当前 TP=%d PP=%d。\n"
            "请先用 tools/checkpoint_util.py 把 checkpoint 重切分到 TP=1/PP=1，\n"
            "或者在训练时就使用单机 TP=1/PP=1（推荐，8 份权重完全相同，无需合并）。"
            % (tp, pp)
        )

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32}[args.dtype]

    # ---- 顶层部件 ----
    emb = get_path(sd, ["model.language_model.embedding.word_embeddings.weight",
                        "language_model.embedding.word_embeddings.weight",
                        "word_embeddings.weight"])
    try:
        out_layer = get_path(sd, ["model.language_model.output_layer.weight",
                                  "language_model.output_layer.weight",
                                  "output_layer.weight"])
        tie = False
    except KeyError:
        out_layer = emb  # tie word embeddings
        tie = True
        print("[!] 没找到 output_layer，按 tie_word_embeddings=True 处理")

    final_norm = get_path(sd, ["model.language_model.encoder.final_layernorm.weight",
                               "language_model.encoder.final_layernorm.weight",
                               "final_layernorm.weight"])
    print("[i] embedding: %s | output_layer: %s | final_norm: %s"
          % (tuple(emb.shape), tuple(out_layer.shape), tuple(final_norm.shape)))

    vocab_rows = emb.shape[0]
    vocab_size = int(args.vocab_size) if args.vocab_size else vocab_rows
    if vocab_size != vocab_rows:
        print("[!] config vocab_size=%d，但 embedding 有 %d 行，已按 embedding 行数修正"
              % (vocab_size, vocab_rows))
        vocab_size = vocab_rows

    out = {}
    out["model.embed_tokens.weight"] = emb.to(dtype).clone()
    out["lm_head.weight"] = out_layer.to(dtype).clone()
    out["model.norm.weight"] = final_norm.to(dtype).clone()

    # ---- 用 qkv 实际形状反推 kv_heads（形状才是真相，args 可能是假信息）----
    # 实测教训：训练脚本传了 --num-query-groups 2，ckpt args 里也记着 2，
    # 但 fused qkv 实际是 6144 行 = (16 + 2*16)*128 ⇒ kv_heads 其实是 16（MHA）。
    # ⇒ 一律不要信 margs.num_query_groups，直接从形状反解：out = (heads + 2*kv) * head_dim
    first_qkv = get_path(sd, layer_key(0, "self_attention.query_key_value.weight"))
    if first_qkv.shape[0] % head_dim != 0:
        raise ValueError("qkv 行数 %d 不能被 head_dim %d 整除，无法反推 kv_heads"
                         % (first_qkv.shape[0], head_dim))
    n_units = first_qkv.shape[0] // head_dim
    rem = n_units - heads
    if rem < 2 or rem % 2 != 0:
        raise ValueError(
            "qkv 形状异常：units=%d, heads=%d ⇒ 剩余 %d 不是合法的 2*kv_heads。"
            "请核对 head_dim(=%d) 是否等于 kv_channels。" % (n_units, heads, rem, head_dim))
    kv_heads = rem // 2
    if heads % kv_heads != 0:
        raise ValueError("heads(%d) 必须能被 kv_heads(%d) 整除" % (heads, kv_heads))
    print("[i] 由 qkv 实际形状反推：kv_heads=%d （ckpt args 里 num_query_groups=%d）" % (kv_heads, qg))
    if kv_heads != qg:
        print("[!] 两者不一致 —— 以形状为准。说明训练时 --num-query-groups=%d 未真正生效，"
              "实际训出来的是 kv_heads=%d 的注意力结构。" % (qg, kv_heads))

    # ---- 逐层 ----
    for i in range(num_layers):
        hf = "model.layers.%d." % i

        out[hf + "input_layernorm.weight"] = get_path(sd, layer_key(i, "input_layernorm.weight")).to(dtype).clone()
        out[hf + "post_attention_layernorm.weight"] = get_path(sd, layer_key(i, "post_attention_layernorm.weight")).to(dtype).clone()

        qkv = get_path(sd, layer_key(i, "self_attention.query_key_value.weight"))
        q, k, v = split_qkv_gqa(qkv, heads, kv_heads, head_dim)
        out[hf + "self_attn.q_proj.weight"] = q.to(dtype).clone()
        out[hf + "self_attn.k_proj.weight"] = k.to(dtype).clone()
        out[hf + "self_attn.v_proj.weight"] = v.to(dtype).clone()

        out[hf + "self_attn.o_proj.weight"] = get_path(sd, layer_key(i, "self_attention.dense.weight")).to(dtype).clone()

        # SwiGLU：Megatron 是 silu(前半) * 后半 ⇒ 前半 gate，后半 up
        w = get_path(sd, layer_key(i, "mlp.dense_h_to_4h.weight"))
        if w.shape[0] != 2 * ffn:
            raise ValueError("dense_h_to_4h 行数 %d != 2*ffn %d" % (w.shape[0], 2 * ffn))
        out[hf + "mlp.gate_proj.weight"] = w[:ffn, :].to(dtype).clone()
        out[hf + "mlp.up_proj.weight"] = w[ffn:, :].to(dtype).clone()
        out[hf + "mlp.down_proj.weight"] = get_path(sd, layer_key(i, "mlp.dense_4h_to_h.weight")).to(dtype).clone()

        if (i + 1) % 8 == 0 or i == 0:
            print("    已转换 %d/%d 层" % (i + 1, num_layers))

    # ---- config.json（Qwen2）----
    config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_hidden_layers": num_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_position_embeddings": args.max_position_embeddings,
        "rope_theta": args.rope_theta,
        "rms_norm_eps": args.rms_norm_eps,
        # 训练脚本带 --disable-bias-linear ⇒ 权重里没有 qkv bias。
        # 写成 true 会让 HF/vLLM 随机初始化一份 bias，是静默错误。
        "qkv_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": bool(tie),
        "use_cache": True,
        "torch_dtype": args.dtype,
        "bos_token_id": args.eos_token_id,
        "eos_token_id": args.eos_token_id,
        "transformers_version": "4.40.0",
    }

    # ---- 保存 ----
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    saved_as = None
    if args.save_safetensors:
        try:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in out.items()},
                      os.path.join(args.save_path, "model.safetensors"))
            saved_as = "model.safetensors"
        except ImportError:
            print("[!] 没装 safetensors，退回 pytorch_model.bin")
    if saved_as is None:
        torch.save(out, os.path.join(args.save_path, "pytorch_model.bin"))
        saved_as = "pytorch_model.bin"

    # ---- tokenizer ----
    if args.tokenizer_path and os.path.isdir(args.tokenizer_path):
        n = 0
        for f in os.listdir(args.tokenizer_path):
            low = f.lower()
            if (low.startswith("tokenizer") or low.startswith("vocab")
                    or low.startswith("merges") or low.endswith(".tiktoken")
                    or low in ("special_tokens_map.json", "added_tokens.json")):
                src = os.path.join(args.tokenizer_path, f)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(args.save_path, f))
                    n += 1
        print("[i] 拷贝了 %d 个 tokenizer 文件" % n)

    # ---- 汇总 ----
    total = sum(v.numel() for v in out.values())
    print("\n=== 转换完成 ===")
    print("    输出目录 : %s" % args.save_path)
    print("    权重文件 : %s" % saved_as)
    print("    tensor 数 : %d" % len(out))
    print("    参数量   : %.3f B" % (total / 1e9))
    print("    dtype    : %s" % args.dtype)
    print("    config   : vocab=%d layers=%d hidden=%d heads=%d kv_heads=%d rope_theta=%s"
          % (vocab_size, num_layers, hidden, heads, kv_heads, args.rope_theta))
    print("\n[!] 两个必须人工核对的点：")
    print("    1) rope_theta=%s 必须等于训练时的 --rotary-emb-base（该 fork 默认 10000）"
          % args.rope_theta)
    print("    2) qkv_bias=False 必须成立（训练脚本带 --disable-bias-linear）")


def main():
    args = parse_args()
    if args.print_structure and not args.save_path:
        convert(args)
        return
    if not args.save_path:
        print("错误：需要 --save-path，或加 --print-structure 只做探查")
        sys.exit(1)
    convert(args)


if __name__ == "__main__":
    main()
