# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# !/usr/bin/env python3
"""SFT GLM"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.model import GLMModel
from megatron.sft_training import SFTTrainer
from megatron.core.enums import ModelType
from megatron.utils import average_losses_across_data_parallel_group
from megatron.data.glm_data import build_mask_matrix
from megatron.data.sft_glm_data import build_glm_sft_datasets, CustomizationDataset



try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        """record"""
        return fn


def model_provider(pre_process=True,
                   post_process=True):
    """Build the model."""
    print_rank_0('building GLM model ...')
    model = GLMModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               set_loss_mask=False,
                               mem_length=None):
    """get masks and position ids"""
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if mem_length:
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = torch.tril(torch.triu(attention_mask, 1 - seq_length + mem_length), mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if set_loss_mask:
        loss_mask[data == eod_token] = 0.0
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

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

    return attention_mask, loss_mask, position_ids


def process_data(data):
    """process data."""
    args = get_args()
    tokenizer = get_tokenizer()
    # Items and their type.
    keys = ['text', 'loss_mask']
    if args.block_lm:
        keys += ['target', 'attention_mask', 'position_id']
    # if not args.finetune and args.deepspeed:
    #     keys.append('task_type')
    datatype = torch.int64
    print_rank_0(f"====== data:{data} ==========")
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    print_rank_0(f"====== data_b:{data_b} ==========")

    tokens = data_b['text'].long()
    labels = data_b['target'].long()
    attention_mask = data_b['attention_mask'].long()
    attention_mask = build_mask_matrix(attention_mask, tokens.size(0),
                                       tokens.size(1))
    loss_mask = data_b['loss_mask'].float()
    position_ids = data_b['position_id'].long()

    if not args.block_lm:
        attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
            tokens,
            args.eod_token,
            args.reset_position_ids,
            args.reset_attention_mask,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            mem_length=args.mem_length,
            set_loss_mask=not args.transformer_xl)
        # Convert
        if args.fp16:
            attention_mask = attention_mask.half()
    attention_mask = attention_mask.to(torch.bool)

    return [tokens, labels, loss_mask, attention_mask, position_ids]


def get_batch(data_iterator):
    """Generate a batch"""
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    return process_data(data)


def loss_func(loss_mask, output_tensor):
    """loss func."""
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    print("the loss is {} and lm loss is {}".format(loss, averaged_loss[0]))

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    args = get_args()
    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor, logits = model(tokens, position_ids, attention_mask,
                          labels=labels)
    if args.label_smoothing > 0.0:
        epsilon = args.label_smoothing
        smooth_loss = -torch.nn.functional.log_softmax(logits, dim=-1).mean(dim=-1)
        output_tensor = (1 - epsilon) * output_tensor + epsilon * smooth_loss

    return output_tensor, partial(loss_func, loss_mask)


#
def sft_glm_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()
    train_ds, valid_ds, test_ds = None, None, None
    print_rank_0('> building train, validation, and test datasets for GLM SFT ...')
    assert args.data_path is not None

    data_path = ""
    if isinstance(args.data_path, list) and len(args.data_path) > 0:
        data_path = args.data_path[0]
    elif isinstance(args.data_path, str):
        data_path = args.data_path
    #train_ds = build_glm_sft_datasets(data_path=data_path)
    train_ds = CustomizationDataset(args, data_path, tokenizer)
    if args.valid_data_path:
        valid_ds = build_glm_sft_datasets(data_path=args.valid_data_path)
    if args.test_data_path:
        test_ds = build_glm_sft_datasets(data_path=args.test_data_path)
    print_rank_0("> finished creating GLM SFT datasets ...")
    return train_ds, valid_ds, test_ds


@record
def main():
    """main entry"""
    trainer = SFTTrainer(
        sft_glm_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GLMChineseTokenizer'}
    )
    trainer.train()


if __name__ == "__main__":
    main()
