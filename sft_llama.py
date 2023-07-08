# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# !/usr/bin/env python3
"""SFT LLaMA"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.sft_llama_data import build_sft_llama_datasets
from megatron.model import LLaMAModel
from megatron.utils import average_losses_across_data_parallel_group
from megatron.sft_training import SFTTrainer
from typing import Optional
from torch import nn

IGNORE_INDEX = -100


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building LLaMA model ...')
    model = LLaMAModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_masks_position_ids(input_ids, device):
    """Generate attention mask."""
    tokenizer = get_tokenizer()
    batch_size, seq_length = input_ids.shape

    attention_mask = torch.tril(torch.ones(
        (batch_size, seq_length, seq_length), device=device)).view(
        batch_size, 1, seq_length, seq_length)

    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    position_ids = position_ids.clone()

    for b in range(batch_size):
        # Find indecies where EOD token is.
        eod_index = position_ids[b, input_ids[b] == tokenizer.eos_token_id]
        # Detach indecies from positions if going to modify positions.
        eod_index = eod_index.clone()
        # Loop through EOD indecies:

        prev_index = 1
        for j in range(eod_index.size()[0]):
            i = eod_index[j]
            # Mask attention loss.
            attention_mask[b, 0, (i + 1):, :seq_length-(i + 1)] = 1
            # Reset positions.
            position_ids[b, (i + 1):] -= (i + 1 + prev_index)
            prev_index = i + 1
    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, position_ids


def get_position_ids(input_ids, mask_positions, device):
    """Generate position ids."""
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

    prev_index = 1
    for i, mask_position in enumerate(mask_positions):
        #position_ids[i, :mask_position[0]+1] = mask_position[0]
        position_ids[i, mask_position[1]+2:] -= mask_position[1]+2 + prev_index
        prev_index += 1

    return position_ids


def get_masks(input_ids, mask_positions, device):
    """Generate attention mask."""

    tokenizer = get_tokenizer()
    batch_size, seq_length = input_ids.shape
    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
    attention_mask.tril_()
    for i, mask_position in enumerate(mask_positions):
        #attention_mask[i, :, :mask_position[0]+1] = 1
        attention_mask[i, :, mask_position[1]+2:] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    return attention_mask


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    return combined_attention_mask


def make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def get_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""
    tokenizer = get_tokenizer()
    args = get_args()

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # attention mask
    past_key_values_length = 0

    attention_mask = torch.ones(
        (micro_batch_size, seq_length), dtype=torch.bool, device=data.device
    )

    embed_tokens = nn.Embedding(tokenizer.vocab_size, args.hidden_size, tokenizer.pad_token_id, device=data.device)

    inputs_embeds = embed_tokens(data)

    attention_mask = prepare_decoder_attention_mask(
        attention_mask, (micro_batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    # Position ids.
    device = data.device
    position_ids = torch.arange(
        past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

    return attention_mask, position_ids


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['input_ids', 'labels', 'loss_mask', 'attention_mask', 'ignore_lengths']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    print_rank_0(f"====== data:{data} ==========")
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    print_rank_0(f"====== data_b:{data_b} ==========")

    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    mask_positions = data_b['ignore_lengths'].long()
    loss_mask = data_b['loss_mask'].float()
    print_rank_0(f"the data_b loss_mask is {loss_mask}")
    #attention_mask = data_b['attention_mask'].bool()

    #attention_mask = get_masks(tokens, mask_positions, tokens.device)
    #print_rank_0(f"the attention mask is {attention_mask}")
    # Position ids.
    #position_ids = get_position_ids(tokens, mask_positions, tokens.device)
    #print_rank_0(f"the position id is {position_ids}")

    attention_mask, position_ids = get_masks_position_ids(tokens, tokens.device)

    #attention_mask, position_ids = get_masks_and_position_ids(tokens)

    labels = torch.cat((labels[:, 1:], labels[:, :1]), dim=1)
    loss_mask[:, -1] = 0

    return (tokens, labels, loss_mask), (attention_mask, position_ids)


def loss_func(loss_mask, output_tensor):
    """Loss func."""
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    print("the loss is {} and lm loss is {}".format(loss, averaged_loss[0]))

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    (tokens, labels, loss_mask), (attention_mask, position_ids) = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def sft_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    train_ds, valid_ds, test_ds = None, None, None
    print_rank_0('> building train, validation, and test datasets '
                 'for LLaMA SFT ...')
    assert args.data_path is not None

    data_path = ""
    if isinstance(args.data_path, list) and len(args.data_path) > 0:
        data_path = args.data_path[0]
    elif isinstance(args.data_path, str):
        data_path = args.data_path
    train_ds = build_sft_llama_datasets(data_path=data_path)
    if args.valid_data_path:
        valid_ds = build_sft_llama_datasets(data_path=args.valid_data_path)
    if args.test_data_path:
        test_ds = build_sft_llama_datasets(data_path=args.test_data_path)

    print_rank_0("> finished creating sft LLaMA datasets ...")

    return train_ds, valid_ds, test_ds


def main():
    """main entry"""
    trainer = SFTTrainer(
        sft_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'LLaMASFTTokenizerHF'}
    )
    trainer.train()


if __name__ == "__main__":
    main()
