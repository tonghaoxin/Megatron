#!/usr/bin/python
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate LLAMA"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import LLaMAModel
from megatron.training import get_model
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
from megatron.accelerator import get_accelerator
import torch

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building LLaMA model ...')
    model = LLaMAModel(
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    return parser

if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'LLaMASentencePieceTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # ★昆仑芯 P800 一次性推理模式（2026-10-06 新增）
    #   背景：Flask 的 put() 是在【工作线程】里执行的，它连续发起三次
    #     torch.distributed.broadcast（send_do_generate → broadcast_float_list
    #     → tokenize_prompts 里的 broadcast_tensor/broadcast_int_list）。
    #   在 XPU + XCCL 上观察到的症状是：非 0 号 rank 拿到的 prompt 张量是
    #     torch.empty 的未初始化显存（id 量级 1e14~8e18，连 prompt 那几位都是垃圾），
    #   而 rank0 自己那条流水线（rank 0/4/8/12）再也没返回，HTTP 永久挂住。
    #   本分支让「广播 + 生成」全部发生在【主线程】，用于一举判定是否就是
    #   「非主线程发起集合通信」的问题；若能跑通，它本身就是可用的推理入口。
    _once_prompt = os.environ.get("INF_ONCE_PROMPT", "")
    if _once_prompt:
        _tok_gen = int(os.environ.get("INF_ONCE_TOKENS", "32"))
        _top_k = int(os.environ.get("INF_ONCE_TOP_K", "1"))
        torch.distributed.barrier()
        if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            print("[ONCE] prompt=%r tokens_to_generate=%d top_k=%d" %
                  (_once_prompt, _tok_gen, _top_k), flush=True)
            _out = generate_and_post_process(
                model,
                prompts=[_once_prompt],
                tokens_to_generate=_tok_gen,
                top_k_sampling=_top_k,
                add_BOS=False,
                use_eod_token_for_early_termination=True)
            print("[ONCE] RESULT >>> " + repr(_out[0]), flush=True)
        else:
            generate_and_post_process(model=model)
        torch.distributed.barrier()
        print("[ONCE] DONE rank=%d" % torch.distributed.get_rank(), flush=True)
        sys.exit(0)

    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0")

    while True:
        choice = get_accelerator().LongTensor(1)
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            try:
                generate_and_post_process(model=model)
            except ValueError as ve:
                pass
        elif choice[0].item() == 1:
            try:
                beam_search_and_post_process(model)
            except ValueError as ve:
                pass
