#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Dataloaders."""

import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from megatron import get_args
from megatron.core import mpu
from megatron.global_vars import get_tokenizer
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizer, BatchEncoding
from transformers.utils import is_tf_tensor, is_torch_tensor, to_py_obj
from .collator import ConstructBlockStrategy
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Any, Mapping
from torch.utils.data.dataloader import default_collate


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None:
        return None

    args = get_args()
    tokenizer = get_tokenizer()

    data_collator = None

    if args.seq2seq_data_collator or args.sft_collator:
        data_collator = DataCollatorForSeq2Seq(
            get_tokenizer(),
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=False
        )
    elif args.custom_collator:
        data_collator = my_collate
    elif args.block_lm:
        data_collator = ConstructBlockStrategy(args, tokenizer, args.seq_length, bert_prob=args.bert_prob,
                                               gap_sentence_prob=args.gap_sentence_prob,
                                               gap_sentence_ratio=args.gap_sentence_ratio,
                                               gpt_infill_prob=args.gpt_infill_prob,
                                               average_block_length=args.avg_block_length,
                                               gpt_min_ratio=args.gpt_min_ratio,
                                               block_mask_prob=args.block_mask_prob,
                                               context_mask_ratio=args.context_mask_ratio,
                                               short_seq_prob=args.short_seq_prob,
                                               single_span_prob=args.single_span_prob,
                                               shuffle_blocks=not args.no_shuffle_block,
                                               block_position_encoding=not args.no_block_position,
                                               sentinel_token=args.sentinel_token,
                                               encoder_decoder=args.encoder_decoder,
                                               task_mask=args.task_mask, random_position=args.random_position,
                                               masked_lm=args.masked_lm).construct_blocks

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=mpu.get_data_parallel_rank(),
            data_parallel_size=mpu.get_data_parallel_world_size(),
            data_sharding=args.data_sharding)
    else:
        raise Exception('{} dataloader type is not supported.'.format(
            args.dataloader_type))

    # Torch dataloader.
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )


@dataclass
class CustomDataCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    """
    model: Optional[Any] = None
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if isinstance(features, (list, tuple)) and isinstance(features[0], Mapping):
            features = {key: [example[key] for example in features] for key in features[0].keys()}

        batch_outputs = {}
        required_input = features['text']
        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            # first_element might be an empty list/tuple in some edge cases so we grab the first non empty element.
            for item in required_input:
                if len(item) != 0:
                    first_element = item[0]
                    break
        # At this state, if `first_element` is still a list/tuple, it's an empty one so there is nothing to do.
        if not isinstance(first_element, (int, list, tuple)):
            if is_tf_tensor(first_element):
                return_tensors = "tf" if return_tensors is None else return_tensors
            elif is_torch_tensor(first_element):
                return_tensors = "pt" if return_tensors is None else return_tensors
            elif isinstance(first_element, np.ndarray):
                return_tensors = "np" if return_tensors is None else return_tensors
            else:
                raise ValueError(
                    f"type of {first_element} unknown: {type(first_element)}. "
                    "Should be one of a python, numpy, pytorch or tensorflow object."
                )

            for key, value in features.items():
                features[key] = to_py_obj(value)

        if required_input and not isinstance(required_input[0], (list, tuple)):
            return BatchEncoding(features, tensor_type=return_tensors)
        batch_size = required_input.shape[1]
        for i in range(batch_size):
            outputs = {k: v[i] for k, v in features.items()}
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
        return BatchEncoding(batch_outputs, tensor_type=return_tensors)


def my_collate(batch):
    new_batch = [{key: value for key, value in sample.items() if key != 'uid'} for sample in batch]
    text_list = [sample['text'] for sample in batch]

    def pad_choice_dim(data, choice_num):
        if len(data) < choice_num:
            data = np.concatenate([data] + [data[0:1]] * (choice_num - len(data)))
        return data

    if len(text_list[0].shape) == 2:
        choice_nums = list(map(len, text_list))
        max_choice_num = max(choice_nums)
        for i, sample in enumerate(new_batch):
            for key, value in sample.items():
                if key != 'label':
                    sample[key] = pad_choice_dim(value, max_choice_num)
                else:
                    sample[key] = value
            sample['loss_mask'] = np.array([1] * choice_nums[i] + [0] * (max_choice_num - choice_nums[i]),
                                           dtype=np.int64)

    if 'dec_text' in new_batch[0]:
        choice_nums = [len(sample['dec_text']) for sample in new_batch]
        if choice_nums.count(choice_nums[0]) != len(choice_nums):
            max_choice_num = max(choice_nums)
            for i, sample in enumerate(new_batch):
                for key, value in sample.items():
                    if key.startswith('dec_'):
                        sample[key] = pad_choice_dim(value, max_choice_num)
                sample['loss_mask'] = np.array([1] * choice_nums[i] + [0] * (max_choice_num - choice_nums[i]),
                                               dtype=np.int64)

    new_batch = default_collate(new_batch)
    if 'uid' in batch[0]:
        uid_list = [sample['uid'] for sample in batch]
        new_batch['uid'] = uid_list
    return new_batch


class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.seed
        self.curr_seed = args.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                          * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                               * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
