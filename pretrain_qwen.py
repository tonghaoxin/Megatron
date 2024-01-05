# !/usr/bin/env python3
"""pretrain_qwen."""

from functools import partial
import torch
import os

from megatron.core.enums import ModelType
from megatron.utils import get_ltor_masks_and_position_ids
from megatron import get_args
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.utils import average_losses_across_data_parallel_group

from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import QwenModel
from megatron import get_tokenizer
from megatron.training import pretrain

def model_provider(pre_process=True, post_process=True):
    """get model
    
    Args:
        pre_process (bool): do pre_process or not
        post_process (bool): do post_process or not
    
    Returns:
        torch.nn.Module: return module
    """
    model = QwenModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()
    datatype = torch.int64

    keys = ['text']
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    tokens_ = data_b['text'].long()

    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.pad_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        True)

    return (tokens, labels, loss_mask), (attention_mask, position_ids)


def loss_func(loss_mask, output_tensor):
    """
    loss func
    
    Args:
        loss_mask (Tensor): loss mask
        output_tensor (Tensor): output tensor
    
    Returns:
        Tuple:
            - Tensor: Tensor
            - Dict: Dict
    
    """
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

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path)

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'QwenTokenizer'})

