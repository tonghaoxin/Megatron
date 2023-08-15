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

"""Pretrain ChatGLM2_6B"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.model import GLMModel
from megatron.training import pretrain
from megatron.core.enums import ModelType
from megatron.utils import average_losses_across_data_parallel_group
#from megatron.data.chatglm6b_data import build_train_valid_test_datasets
from megatron.data.chatglm6b_data import build_train_valid_test_datasets


try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        """record"""
        return fn



def get_masks(input_ids, past_key_values, padding_mask=None):
    batch_size, seq_length = input_ids.shape
    full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
    full_attention_mask.tril_()
    past_length = 0
    if past_key_values:
        past_length = past_key_values[0][0].shape[0]
    if past_length:
        full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                    device=input_ids.device), full_attention_mask), dim=-1)
    if padding_mask is not None:
        full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
    if not past_length and padding_mask is not None:
        full_attention_mask -= padding_mask.unsqueeze(-1) - 1
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask

def get_position_ids(input_ids, device):
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return position_ids


def model_provider(pre_process=True,
                   post_process=True):
    """Build the model."""

    print_rank_0('building GLM2 model ...')

    model = GLMModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    return model


def process_data(data):
    """process data."""
    args = get_args()
    tokenizer = get_tokenizer()
    # Items and their type.
    keys = ['input_ids', 'loss_mask', 'labels']
    datatype = torch.int64

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    loss_mask = data_b['loss_mask'].float()
    attention_mask = get_masks(tokens, None, None)
    position_ids = get_position_ids(tokens, tokens.device)

    labels = torch.cat((labels[:, 1:], labels[:, :1]), dim=1)
    loss_mask[:, -1] = 0

    return (tokens, labels, loss_mask), (attention_mask, position_ids)


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
    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    (tokens, labels, loss_mask), (attention_mask, position_ids) = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor[0], partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for ChatGLM2-6b ...')
    # Option 1 of data loading using --data-path

    if args.data_path or args.multitask_data_path:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            length_per_sample=args.length_per_sample,
            aggregated_samples_per_sequence=args.aggregated_samples_per_sequence,
            args=args)
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        raise NotImplementedError()
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating ChatGLM datasets ...")
    return train_ds, valid_ds, test_ds


@record
def main():
    """main"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    #args = parser.parse_args()
    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step,
             args_defaults={'tokenizer_type': 'ChatGLM2Tokenizer'})

if __name__ == "__main__":
    main()

