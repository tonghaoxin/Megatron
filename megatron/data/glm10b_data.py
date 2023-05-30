# !/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""parses arguments and preps data loader"""

import os
import copy
import random
import numpy as np
import time
import torch
import torch.utils.data
from torch.utils import data
from megatron import get_args, print_rank_0
from itertools import accumulate
from bisect import bisect_right
from operator import itemgetter

from megatron import get_tokenizer
from megatron.core import mpu
from . import corpora
from .lazy_loader import exists_lazy, LazyWriter, LazyLoader, exists_scatter, get_scatter_path


def build_train_valid_test_datasets(
    data_prefix, splits_string, train_valid_test_num_samples, seq_length,
    length_per_sample, aggregated_samples_per_sequence, args
):
    args = get_args()
    tokenizer = get_tokenizer()

    data_set_args = {
        'name': args.data_name,
        'path': data_prefix,
        'seq_length': args.seq_length,
        'label_key': 'label',
        'ds_type': 'Block' if args.block_lm else 'GPT2',
        'split': args.split,
        'tokenizer': tokenizer,
        'data_parallel_rank': mpu.get_data_parallel_rank(),
        'no_lazy_loader': True,
        'pre_tokenize': True,
    }

    train, valid, test = None, None, None
    train = make_dataset(**data_set_args)
    train, valid, test = train

    return train, valid, test


class BlockDataset(data.Dataset):
    def __init__(self, ds, tokenizer,
                 max_seq_len=1024,
                 sample_across_doc=True,
                 non_sentence_start=0.0, filter_english=False, **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.sample_across_doc = sample_across_doc
        self.non_sentence_start = non_sentence_start
        self.filter_english = filter_english
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        if self.filter_english:
            import fasttext
            self.model = fasttext.load_model('/mnt/lid.176.bin')
            print_rank_0("Load language detection model")
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.is_lazy:
            lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
        else:
            lens = np.array([len(d['text']) if isinstance(d, dict) else len(d) for d in self.ds])
        self.total_len = np.sum(lens)
        print_rank_0(
            f"Dataset document count {len(lens)}, token count {self.total_len}, non sentence start{self.non_sentence_start}")
        self.weighting = list(accumulate(lens))

    def get_weighted_samples(self, np_rng):
        while True:
            idx = np_rng.randint(self.total_len)
            data_idx = bisect_right(self.weighting, idx)
            tokens, loss_mask = self.getidx(data_idx)
            if self.filter_english:
                text = self.tokenizer.DecodeIds(tokens[:1024])
                lang = self.model.predict(text.replace('\n', ''))[0][0]
                if lang == '__label__en':
                    break
            else:
                break
        return tokens, loss_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        tokens, loss_mask = self.get_weighted_samples(rng)
        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len + 1

        # randomly choose a position for start
        if tokens_to_strip > 0:
            move_count = 0
            strip_left_tokens = rng.randint(tokens_to_strip)
            if rng.random() > self.non_sentence_start:
                if rng.random() < 0.5:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens > 0 and not self.contains_sentence_end(
                            tokens[strip_left_tokens - 1]):
                        strip_left_tokens -= 1
                        move_count += 1
                else:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens < len(
                            tokens) and not self.contains_sentence_end(tokens[strip_left_tokens - 1]):
                        strip_left_tokens += 1
                        move_count += 1
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens[strip_left_tokens:]
            loss_mask = [0] + loss_mask[strip_left_tokens:]
            if len(tokens) == 2 and tokens[1] == self.tokenizer.get_command('eos').Id:
                tokens, loss_mask = [], []
            tokens, loss_mask = self.right_strip_seq(tokens, loss_mask, self.max_seq_len)
        else:
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens
            loss_mask = [0] + loss_mask
            # Sample multiple documents
            if self.sample_across_doc:
                while len(tokens) < self.max_seq_len:
                    new_tokens, new_loss_mask = self.get_weighted_samples(rng)
                    new_tokens = [self.tokenizer.get_command('ENC').Id] + new_tokens
                    new_loss_mask = [0] + new_loss_mask
                    is_last = len(new_tokens) >= self.max_seq_len - len(tokens)
                    new_tokens, new_loss_mask = self.right_strip_seq(new_tokens, new_loss_mask,
                                                                     self.max_seq_len - len(tokens))
                    tokens += new_tokens
                    loss_mask += new_loss_mask
                    if is_last:
                        break
        return {'text': np.array(tokens), "loss_mask": np.array(loss_mask)}

    def right_strip_seq(self, tokens, loss_mask, seq_length):
        strip_right_tokens = len(tokens) - seq_length
        if strip_right_tokens > 0:
            while strip_right_tokens < len(tokens) - 1 and not self.contains_sentence_end(
                    tokens[-strip_right_tokens - 1]):
                strip_right_tokens += 1
            if len(tokens) - strip_right_tokens < seq_length // 2:
                strip_right_tokens = len(tokens) - seq_length
            tokens = tokens[:-strip_right_tokens]
            loss_mask = loss_mask[:-strip_right_tokens]
        return tokens, loss_mask

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.get_command('eos').Id]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.get_command('pad').Id if pad_id is None else pad_id] * (num_pad_tokens)
        return seq

    # TODO: rewrite this function for chinese
    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        if '\n' in tok:
            return True
        return False


class GPT2Dataset(data.Dataset):

    def __init__(self, ds, tokenizer,
                 max_seq_len=1024,
                 num_samples=None,
                 weighted=True,
                 sample_across_doc=True,
                 random_across_doc_sampling=True,
                 sentence_start=False, **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = num_samples
        if num_samples is None:
            self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.weighted = weighted
        self.sample_across_doc = sample_across_doc
        self.random_across_doc_sampling = random_across_doc_sampling
        self.sentence_start = sentence_start
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.weighted:
            if self.is_lazy:
                lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
            else:
                lens = np.array([len(d['text']) if isinstance(d, dict)
                                 else len(d) for d in self.ds])
            self.total_len = np.sum(lens)
            print_rank_0(f"Dataset document count {len(lens)}, token count {self.total_len}")
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None

    def get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        data_idx = self.get_weighted_samples(rng)
        #        data_idx = rng.choice(self.ds_len, p=self.weighting)
        tokens, loss_mask = self.getidx(data_idx)

        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len - 1

        # randomly choose a position for start
        if tokens_to_strip > 0:
            strip_left_tokens = rng.randint(tokens_to_strip + 1)
            tokens = tokens[strip_left_tokens:]
            loss_mask = loss_mask[strip_left_tokens:]
            # if self.sentence_start:
            #     token_copy = list(tokens)
            #     not_done = True
            #     while (len(token_copy) > 0) and not_done:
            #         tok = token_copy.pop(0)
            #         if self.contains_sentence_end(tok):
            #             tokens = token_copy
            #             not_done = False
            strip_right_rokens = len(tokens) - self.max_seq_len - 1
            if strip_right_rokens > 0:
                tokens = tokens[:-strip_right_rokens]
                loss_mask = loss_mask[:-strip_right_rokens]
        # Sample multiple documents
        if self.sample_across_doc:
            while (len(tokens) < (self.max_seq_len + 1)):
                if self.random_across_doc_sampling:
                    data_idx = self.get_weighted_samples(rng)
                else:
                    data_idx = (data_idx + 1) % self.ds_len
                new_tokens, new_loss_mask = self.getidx(data_idx)
                tokens += new_tokens
                loss_mask += new_loss_mask
            tokens = tokens[:(self.max_seq_len + 1)]
            loss_mask = loss_mask[:(self.max_seq_len + 1)]

        tokens = self.pad_seq(tokens)
        loss_mask = self.pad_seq(loss_mask, pad_id=0)
        return {'text': np.array(tokens), "loss_mask": np.array(loss_mask)}

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.get_command('eos').Id]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len + 1
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.get_command('pad').Id if pad_id is None else pad_id] * (num_pad_tokens)
        return seq

    # TODO: rewrite this function for chinese
    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        return False


class ConcatDataset(data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.is_lazy = sum([isinstance(ds, LazyLoader) or (hasattr(ds, 'is_lazy') and ds.is_lazy) for ds in
                            self.datasets]) == len(self.datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._X = None
        self._Y = None
        self._lens = None

    def get_text_len(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_text_len(sample_idx)

    def SetTokenizer(self, tokenizer):
        for ds in self.datasets:
            ds.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.datasets[0].GetTokenizer()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def lens(self):
        if self._lens is None:
            self._lens = []
            if self.is_lazy:
                for data in self.datasets:
                    self._lens.extend(data.lens)
            else:
                for data in self.datasets:
                    self._lens.extend([len(d['text']) if isinstance(d, dict) else len(d) for d in data])
        return self._lens

    @property
    def X(self):
        if self._X is None:
            self._X = []
            for data in self.datasets:
                self._X.extend(data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = []
            for data in self.datasets:
                self._Y.extend(list(data.Y))
            self._Y = np.array(self._Y)
        return self._Y


class SplitDataset(data.Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Purpose: useful to index into existing datasets, possibly
    large-scale datasets as the subindexing operation is done in an
    on-the-fly manner.
    Arguments:
        ds (Dataset or array-like): List of datasets to be subindexed
        split_inds (1D array-like): List of indices part of subset
    """

    def __init__(self, ds, split_inds, **kwargs):
        self.split_inds = list(split_inds)
        self.wrapped_data = ds
        self.is_lazy = isinstance(ds, LazyLoader) or (hasattr(ds, 'is_lazy') and ds.is_lazy)
        self._X = None
        self._Y = None

    def __len__(self):
        return len(self.split_inds)

    def get_text_len(self, idx):
        return self.wrapped_data.get_text_len(self.split_inds[idx])

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    def SetTokenizer(self, tokenizer):
        self.wrapped_data.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.wrapped_data.GetTokenizer()

    @property
    def X(self):
        if self._X is None:
            self._X = itemgetter(*self.split_inds)(self.wrapped_data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.array(itemgetter(*self.split_inds)(self.wrapped_data.Y))
        return self._Y

    def __iter__(self):
        for idx in self.split_inds:
            yield self.wrapped_data[idx]


def split_ds(ds, split=None, shuffle=True, save_splits=None, load_splits=None):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
        save_splits: save split indices to file
        load_splits: load split indices from file
    """
    if split is None:
        split = [.8, .2, .0]
    split = get_split(split)
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        rng = np.random.RandomState(1234)
        rng.shuffle(inds)
    if load_splits is not None:
        inds = np.load(load_splits)
        assert len(inds) == ds_len
        print_rank_0(f"Load split indices from {load_splits}")
    elif save_splits is not None:
        if torch.distributed.get_rank() == 0:
            np.save(save_splits, inds)
            print(f"Save split indices to {save_splits}")
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None] * len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len * split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx:start_idx + max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds


def supported_corpus(corpus_name):
    """checks if corpus name is defined in `corpora.py`"""
    return corpus_name in corpora.NAMED_CORPORA


def make_dataset(name, path, seq_length, shuffle=True, split=None, tokenizer=None,
                 sample_one_document=False, pre_tokenize=False, ds_type='', save_splits=None, load_splits=None,
                 save_test_data=None, no_lazy_loader=False, loader_scatter=None, data_parallel_rank=None,
                 filter_english=False, non_sentence_start=0.0, half_lazy_loader=False, **kwargs):
    """function to create datasets+tokenizers for common options"""
    if split is None:
        split = [1.]

    # get one or multiple datasets and concatenate
    if isinstance(path, str):
        ds = get_dataset(name, path, tokenizer=tokenizer, pre_tokenize=pre_tokenize, no_lazy_loader=no_lazy_loader,
                         loader_scatter=loader_scatter, data_parallel_rank=data_parallel_rank,
                         half_lazy_loader=half_lazy_loader)
    else:
        ds = [get_dataset(name, p, tokenizer=tokenizer, pre_tokenize=pre_tokenize, no_lazy_loader=no_lazy_loader,
                          loader_scatter=loader_scatter, data_parallel_rank=data_parallel_rank,
                          half_lazy_loader=half_lazy_loader) for p in path]
        ds = ConcatDataset(ds)

    # Split dataset into train/val/test (and wrap bert dataset)
    def wrap_dataset(dataset):
        if ds_type.lower() == 'gpt2':
            dataset = GPT2Dataset(dataset, tokenizer, max_seq_len=seq_length, sample_across_doc=not sample_one_document)
        elif ds_type.lower() == 'block':
            dataset = BlockDataset(dataset, tokenizer, max_seq_len=seq_length,
                                   sample_across_doc=not sample_one_document, filter_english=filter_english,
                                   non_sentence_start=non_sentence_start)
        return dataset

    ds = split_ds(ds, split, shuffle=shuffle, save_splits=save_splits, load_splits=load_splits)
    ds = [wrap_dataset(d) if d is not None else None for d in ds]
    return ds


def get_dataset(name, path, tokenizer, pre_tokenize, data_parallel_rank, loader_scatter=None, no_lazy_loader=False,
                half_lazy_loader=False):
    """gets dataset object based on keyword args and file at `path`"""
    global_rank = torch.distributed.get_rank()
    if not supported_corpus(name):
        raise NotImplementedError('dataset %s is not supported' % name)
    dataset = corpora.NAMED_CORPORA[name]
    dataset.PATH = path
    if issubclass(dataset, corpora.PromptReader):
        if not (exists_lazy(path, data_type='prompt') and exists_lazy(path, data_type='text')) and not (
                loader_scatter is not None and exists_scatter(path, data_type='prompt',
                                                              scatter_num=loader_scatter) and exists_scatter(path,
                                                                                                             data_type='text',
                                                                                                             scatter_num=loader_scatter)):
            # create cached version of dataset for lazy loading if it doesn't exist
            if global_rank == 0:
                print(f"Creating lazy loader for dataset {name}")
                prompt_writer = LazyWriter(path, data_type='prompt', is_array=pre_tokenize)
                text_writer = LazyWriter(path, data_type='text', is_array=pre_tokenize)
                writers = {'prompt': prompt_writer, 'text': text_writer}
                reader = dataset(writers=writers, tokenizer=tokenizer, tokenize=pre_tokenize)
                reader.process()
                prompt_writer.close()
                text_writer.close()
            else:
                while not os.path.exists(LazyWriter.get_len_path(path, data_type='prompt')):
                    time.sleep(1)
        map_fn = (lambda x: x.tolist()) if pre_tokenize else None
        if loader_scatter is not None:
            if not (exists_scatter(path, data_type='prompt', scatter_num=loader_scatter) and exists_scatter(path,
                                                                                                            data_type='text',
                                                                                                            scatter_num=loader_scatter)):
                if global_rank == 0:
                    print(f"Creating scatter loader for dataset {name}")
                    prompts = LazyLoader(path, data_type='prompt', map_fn=map_fn, mem_map=True,
                                         is_array=pre_tokenize)
                    texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True,
                                       is_array=pre_tokenize)
                    indices = list(range(len(texts)))
                    random.shuffle(indices)
                    segment_length = (len(indices) - 1) // loader_scatter + 1
                    for i in range(loader_scatter):
                        scatter_path = get_scatter_path(path, scatter_rank=i)
                        prompt_writer = LazyWriter(scatter_path, data_type='prompt', is_array=pre_tokenize)
                        text_writer = LazyWriter(scatter_path, data_type='text', is_array=pre_tokenize)
                        for idx in indices[i * segment_length: (i + 1) * segment_length]:
                            prompt_writer.write(prompts[idx])
                            text_writer.write(texts[idx])
                        prompt_writer.close()
                        text_writer.close()
                else:
                    while not (
                            exists_scatter(path, data_type='prompt', scatter_num=loader_scatter) and exists_scatter(
                        path, data_type='text', scatter_num=loader_scatter)):
                        time.sleep(1)
            scatter_path = get_scatter_path(path, scatter_rank=data_parallel_rank % loader_scatter)
            print(f"Rank {global_rank} is using scatter from {scatter_path}")
            prompts = LazyLoader(scatter_path, data_type='prompt', map_fn=map_fn, mem_map=True,
                                 is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
            texts = LazyLoader(scatter_path, data_type='text', map_fn=map_fn, mem_map=True,
                               is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
        else:
            prompts = LazyLoader(path, data_type='prompt', map_fn=map_fn, mem_map=True,
                                 is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
            texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True,
                               is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
        text = corpora.PromptDataset(prompt_loader=prompts, text_loader=texts, tokenizer=tokenizer,
                                     to_tokenize=not pre_tokenize)
        return text
    elif issubclass(dataset, corpora.KeyReader):
        if not (exists_lazy(path, data_type='text') and exists_lazy(path, data_type='mask')):
            # create cached version of dataset for lazy loading if it doesn't exist
            if global_rank == 0:
                text_writer = LazyWriter(path, data_type='text', is_array=pre_tokenize)
                mask_writer = LazyWriter(path, data_type='mask', is_array=True)
                writers = {'mask': mask_writer, 'text': text_writer}
                dataset(writers=writers, tokenizer=tokenizer, tokenize=pre_tokenize)
                mask_writer.close()
                text_writer.close()
            else:
                while not os.path.exists(LazyWriter.get_len_path(path, data_type='mask')):
                    time.sleep(1)
        map_fn = (lambda x: x.tolist()) if pre_tokenize else None
        masks = LazyLoader(path, data_type='mask', map_fn=map_fn, mem_map=True, is_array=True)
        texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True, is_array=pre_tokenize)
        text = corpora.KeyDataset(mask_loader=masks, text_loader=texts, tokenizer=tokenizer,
                                  to_tokenize=not pre_tokenize)
        return text


def get_split(split):
    """
    Get dataset splits from comma separated string list
    """
    splits = []
    if split.find(',') != -1:
        splits = [float(s) for s in split.split(',')]
    elif split.find('/') != -1:
        splits = [float(s) for s in split.split('/')]
    else:
        splits = [float(split)]
    split_total = sum(splits)
    if split_total < 1.:
        splits.append(1 - split_total)
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    final_sum = sum(splits)
    return [s / final_sum for s in splits]

