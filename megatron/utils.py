# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""General utilities."""

import os, sys
from random import random

import torch
from torch.nn.parallel import DistributedDataParallel as torchDDP

from megatron import (
    get_args,
    get_adlr_autoresume,
)
from megatron.core import mpu
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.accelerator import get_accelerator

custom_multi_tensor_applier = get_accelerator().get_operator('custom_multi_tensor_applier')

def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared

def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = get_accelerator().IntTensor([0])
    norm, _ = custom_multi_tensor_applier(
        dummy_overflow_buf,
        [params_data],
        False, # no per-parameter norm
        op='l2norm',
        memory_saving=args().memory_saving
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    if mpu.get_data_parallel_world_size() > 1:
        torch.distributed.all_reduce(averaged_losses,
                                    group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        get_accelerator().memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        get_accelerator().max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        get_accelerator().memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        get_accelerator().max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model,
                                      optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def is_last_rank():
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)

def print_rank(message):
    """ print only on debug rank."""
    args = get_args()
    tp = mpu.get_tensor_model_parallel_rank()
    pp = mpu.get_pipeline_model_parallel_rank()
    dp = mpu.get_data_parallel_rank()
    if tp in args.debug_tp_list \
            and pp in args.debug_pp_list \
            and dp in args.debug_dp_list:
        print(message, flush=True)

def debug(module, stage, **kwargs):
    """
        output:

        ........ 【layer_number】 module_name, shape, value, save_path
    """
    args = get_args()
    title = "........ "
    if hasattr(module, 'layer_number'):
        if module.layer_number in args.debug_layer_list:
            title += f"【{module.layer_number}】"
        else:
            return
    title += f"{module.__class__.__name__}, {stage}, "

    for k, v in kwargs.items():
        if not isinstance(v, torch.Tensor):
            msg = f"{k} {v}"
        else:
            t = torch.flatten(v)
            msg = f"{k} {v.shape}, [{t[0]}, ..., {t[-1]}]"
            if args.debug_save_path is not None:
                filename = os.path.join(args.debug_save_path, f"{k}_{random()}.pt")
                torch.save(v.cpu(), filename)
                msg += ", save to: " + filename
        print_rank(title + msg)

def get_hook_fn(hook_name, save_path=None):
    name_map = {
        "TransformerLanguageModel": ["enc_input_ids", "enc_position_ids", "enc_attn_mask"],
        "Embedding": ["input_ids", "position_ids"],
        "ParallelTransformer": ["hidden_states", "attention_mask"],
        "ParallelTransformerLayer": ["hidden_states", "attention_mask"],
        "ParallelAttention":  ["hidden_states", "attention_mask"],
        "CoreAttention": ["q", "k", 'v', "attention_mask"]
    }
    def hook_fn(module, inputs, outputs):
        """ output:

            >【layer_number】module_name  <hook_name>
                param_name  shape  value
                ...
        """
        layer_number = module.layer_number if hasattr(module, 'layer_number') else None
        module_name = module.__class__.__name__
        flag = ">" if hook_name == "forward" else "<"
        if layer_number is not None:
            print_rank(f"{flag}【{layer_number}】{module_name}  <{hook_name}>")
        else:
            print_rank(f"{flag}【{module_name}】 <{hook_name}>")

        # inputs
        if not isinstance(inputs, tuple):
            inputs = [inputs]
        for idx, _in in enumerate(inputs):
            param_name = name_map[module_name][idx]
            if not isinstance(_in, torch.Tensor):
                print_rank("\t{:20}\t{:40}".format(
                    param_name,
                    str(_in)))
                continue
            t = torch.flatten(_in)
            print_rank("\t{:20}\t{:40}\t{:30}".format(
                    param_name,
                    str(_in.shape),
                    f"[{t[0]}, ... , {t[-1]}]"))
            if save_path is not None:
                if layer_number is not None:
                    filename = f'{hook_name}_layer{layer_number}_{module_name}_{param_name}.pt'
                else:
                    filename = f'{hook_name}_{module_name}_{param_name}.pt'
                torch.save(_in.cpu(), os.path.join(save_path, filename))

        # outputs
        if not isinstance(outputs, tuple):
            outputs = [outputs]
        for idx, _out in enumerate(outputs):
            if not isinstance(_out, torch.Tensor):
                print_rank("\t{:20}\t{:40}".format(
                    f"output{idx}",
                    str(_out)))
                continue
            t = torch.flatten(_out)
            print_rank("\t{:20}\t{:40}\t{:30}".format(
                    f"output{idx}",
                    str(_out.shape),
                    f"[{t[0]}, ... , {t[-1]}]"))
            if save_path is not None:
                if layer_number is not None:
                    filename = f'{hook_name}_layer{layer_number}_{module_name}_output{idx}.pt'
                else:
                    filename = f'{hook_name}_{module_name}_output{idx}.pt'
                torch.save(_out.cpu(), os.path.join(save_path, filename))

    return hook_fn
