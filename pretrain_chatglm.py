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

"""Pretrain ChatGLM_6B"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.model import GLMModel
#, GPTModelPipe
from megatron.training import pretrain
from megatron.core.enums import ModelType
#from megatron.utils import get_ltor_masks_and_position_ids, get_prefix_indices
from megatron.utils import average_losses_across_data_parallel_group
from megatron.data.chatglm6b_data import build_train_valid_test_datasets

from transformers import AutoConfig


try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        """record"""
        return fn

def get_masks(input_ids, device):
    """Generate attention mask."""

    tokenizer = get_tokenizer()
    batch_size, seq_length = input_ids.shape
    context_lengths = [seq.tolist().index(tokenizer.bos_token_id) for seq in input_ids]
    attention_mask = torch.ones((batch_size, seq_length, seq_length), device=device)
    attention_mask.tril_()
    for i, context_length in enumerate(context_lengths):
        attention_mask[i, :, :context_length] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    return attention_mask

def get_position_ids(input_ids, mask_positions, device):
    """Generate position ids."""

    tokenizer = get_tokenizer()
    batch_size, seq_length = input_ids.shape
    context_lengths = [seq.tolist().index(tokenizer.bos_token_id) for seq in input_ids]

    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

    for i, context_length in enumerate(context_lengths):
        position_ids[i, context_length:] = mask_positions[i]
    block_position_ids = [torch.cat((
        torch.zeros(context_length, dtype=torch.long, device=device),
        torch.arange(seq_length - context_length, dtype=torch.long, device=device) + 1
    )) for context_length in context_lengths]
    block_position_ids = torch.stack(block_position_ids, dim=0)

    return torch.stack((position_ids, block_position_ids), dim=1)

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
    attention_mask = get_masks(tokens, tokens.device)
    MASK, gMASK = tokenizer.mask_token_id, tokenizer.gmask_token_id
    mask_positions = []
    for seq in tokens.tolist():
        mask_token = gMASK if gMASK in seq else MASK
        mask_positions.append(seq.index(mask_token))
    position_ids = get_position_ids(tokens, mask_positions, tokens.device)
    return (tokens, labels, loss_mask), (attention_mask, position_ids)


def get_batch(data_iterator):
    """Generate a batch"""
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    return process_data(data)


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    (tokens, labels, loss_mask), (attention_mask, position_ids) = process_data(data)

    if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)



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

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for ChatGLM-6b ...')
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

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds


@record
def main():
    """main"""
    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step,
             args_defaults={'tokenizer_type': 'ChatGLMTokenizer'})

if __name__ == "__main__":
    main()

