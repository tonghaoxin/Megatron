# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GLM-10B"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import get_timers
from megatron.core import mpu, tensor_parallel
from megatron.model import GLMModel
from megatron.training import pretrain
from megatron.core.enums import ModelType
from megatron.utils import average_losses_across_data_parallel_group

from megatron.data.glm_data import build_mask_matrix
from megatron.data.glm10b_data import build_train_valid_test_datasets

import os
import random

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        """record"""
        return fn


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               set_loss_mask=False,
                               mem_length=None):
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


def model_provider(pre_process=True,
                   post_process=True):
    """Build the model."""

    print_rank_0('building GLM 10B model ...')

    model = GLMModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    return model

def process_data(data, args):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['text', 'loss_mask']
    if args.block_lm:
        keys += ['target', 'attention_mask', 'position_id']

    datatype = torch.int64

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    # Unpack.
    if args.block_lm:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].long()
        loss_mask = data_b['loss_mask'].float()
        position_ids = data_b['position_id'].long()
        attention_mask = build_mask_matrix(attention_mask, tokens.size(0),
                                       tokens.size(1))
    else:
        tokens_ = data_b['text'].long()
        loss_mask = data_b['loss_mask'].float()
        labels = tokens_[:, 1:].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
        attention_mask = None

    # Get the masks and postition ids.
    if not args.block_lm:
        tokenizer = get_tokenizer()
        eod_token = tokenizer.get_command('eos').Id
        attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
            tokens,
            eod_token,
            args.reset_position_ids,
            args.reset_attention_mask,
            loss_mask=loss_mask,
            attention_mask=attention_mask)
        # Convert
        if args.fp16:
            attention_mask = attention_mask.half()
    attention_mask = attention_mask.to(torch.bool)
    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    return process_data(data, args)


def loss_func(loss_mask, output_tensor):
    """loss func."""
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
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    (train_ds, val_ds, test_ds) = (None, None, None)

    if args.data_path or args.multitask_data_path:
        train_ds, val_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            length_per_sample=args.length_per_sample,
            aggregated_samples_per_sequence=args.aggregated_samples_per_sequence,
            args=args)
    print_rank_0("> finished creating GLM datasets ...")

    return train_ds, val_ds, test_ds


@record
def main():
    """main"""
    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step,
             args_defaults={'tokenizer_type': 'ChineseSPTokenizer'})

if __name__ == "__main__":
    main()

