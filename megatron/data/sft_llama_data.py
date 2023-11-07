# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# !/usr/bin/env python3
"""SFT data"""
from __future__ import annotations

import torch

from megatron import get_tokenizer, get_args, set_tokenizer
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from megatron.data.dataset_utils import get_datasets_weights_and_num_samples
from megatron.data.sft_data import init_for_sft, preprocess_supervised_dataset, parse_data_args
import os


def build_train_sft_datasets(data_prefix, train_valid_test_num_samples):
    """build online sft dataset"""
    args = get_args()
    data_args = parse_data_args(args.data_args)
    if isinstance(data_args, tuple):
        data_args = data_args[0]

    if len(data_prefix) == 1:
        return build_sft_llama_datasets(data_prefix[0], train_valid_test_num_samples[0], data_args)

    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output
    train_datasets = []
    for i in range(len(prefixes)):
        train_ds = build_sft_llama_datasets(prefixes[i],
                                            datasets_train_valid_test_num_samples[i], data_args)
        train_datasets.append(train_ds)
    if data_args.mix_strategy.startswith("interleave"):
        stopping_strategy = "first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted"
        return interleave_datasets(train_datasets, data_args.interleave_probs, stopping_strategy=stopping_strategy)

    return concatenate_datasets(train_datasets)


def build_sft_llama_datasets(data_path, train_valid_test_num_samples, data_args):
    """build sft dataset from data path and data args"""
    tokenizer = get_tokenizer()
    args = get_args()
    seq_length = args.seq_length
    ignore_index = -100

    dataset_attr = init_for_sft("file", data_args)

    data_files = {"train": os.path.join(data_path)}
    extension = "json"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir="",
        use_auth_token=False,
    )
    dataset = raw_datasets["train"]

    for column_name in ["prompt", "query", "response", "history"]:
        if getattr(dataset_attr, column_name) and getattr(dataset_attr, column_name) != column_name:
            dataset = dataset.rename_column(getattr(dataset_attr, column_name), column_name)
    if dataset_attr.system_prompt:  # add system prompt
        dataset = dataset.add_column("system", [dataset_attr.system_prompt] * len(dataset))

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names_train = list(next(iter(dataset)).keys())
    train_dataset = dataset

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != ignore_index else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    train_dataset = train_dataset.map(
        preprocess_supervised_dataset,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names_train,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )

    print(f"the tokenizer eos token id is {tokenizer.eos_token_id} || "
          f"bos token id is {tokenizer.bos_token_id} || pad token id is {tokenizer.pad_token_id}")
    print_supervised_dataset_example(train_dataset[0])

    train_dataset.set_format(
        type=train_dataset.format["type"], columns=train_dataset.column_names,
        format_kwargs=train_dataset.format["format_kwargs"]
    )

    return train_dataset
