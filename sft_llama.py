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
from megatron.data.sft_llama_data import build_train_sft_datasets
from megatron.model import LLaMAModel
from megatron.utils import average_losses_across_data_parallel_group
from megatron.sft_training import SFTTrainer


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


def get_position_ids(input_ids, device):
    """Generate attention mask."""
    tokenizer = get_tokenizer()
    batch_size, seq_length = input_ids.shape

    """
    attention_mask = torch.tril(torch.ones(
        (batch_size, seq_length, seq_length), device=device)).view(
        batch_size, 1, seq_length, seq_length)
    """

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

        for j in range(eod_index.size()[0]):
            i = eod_index[j]
            # Reset positions.
            position_ids[b, (i + 1):] = i + 1

    return position_ids


def get_masks(input_ids, device):
    """Generate attention mask."""

    tokenizer = get_tokenizer()
    batch_size, seq_length = input_ids.shape
    context_lengths = [seq.tolist().index(tokenizer.eos_token_id) for seq in input_ids]
    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
    attention_mask.tril_()
    for i, context_length in enumerate(context_lengths):
        attention_mask[i, :, :context_length] = 1
        attention_mask[i, :, context_length:] = 0
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    return attention_mask


def get_batch(data_iterator):
    """Generate a batch"""

    # Items and their type.
    keys = ['input_ids', 'labels', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    if data_b['input_ids'].ndim > 2:
        data_b['input_ids'].squeeze_(0)
    if data_b['labels'].ndim > 2:
        data_b['labels'].squeeze_(0)
    if data_b['loss_mask'].ndim > 2:
        data_b['loss_mask'].squeeze_(0)
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    loss_mask = data_b['loss_mask'].float()

    position_ids = get_position_ids(tokens, tokens.device)
    attention_mask = get_masks(tokens, tokens.device)

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

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
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

    if args.enable_prep_data:
        from megatron.data.sft_dataset import build_sft_train_datasets
        return build_sft_train_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup),
            train_data_prefix=args.train_data_path,
            valid_data_prefix=args.valid_data_path,
            test_data_prefix=args.test_data_path
        )

    train_ds = build_train_sft_datasets(data_prefix=args.data_path,
                                        train_valid_test_num_samples=train_val_test_num_samples)
    if args.valid_data_path:
        valid_ds = build_train_sft_datasets(data_prefix=args.valid_data_path,
                                            train_valid_test_num_samples=train_val_test_num_samples)
    if args.test_data_path:
        test_ds = build_train_sft_datasets(data_prefix=args.test_data_path,
                                           train_valid_test_num_samples=train_val_test_num_samples)

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
