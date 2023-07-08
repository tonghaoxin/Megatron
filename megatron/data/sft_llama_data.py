# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# !/usr/bin/env python3
"""SFT data"""
from __future__ import annotations

import torch

from megatron import get_tokenizer, get_args, set_tokenizer
from datasets import load_dataset
import os
import copy


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

FILTER_BY_WORDS = [
    "as a language model",
    "as an AI language model",
    "As a large language model",
    "As an AI ",
    "an AI language model you don't have",
    "As an AI language model, I cannot",
    "As an AI language model, I do not",
    "As an AI language model, I am not able",
    "As an AI language model, I don't have personal",
    "I am an AI language model and do not",
    "As an AI language model, I don't have",
    "As an AI language model, I am only able",
    "AI language model and I do not",
    "As an AI language model, I cannot modify",
    "As an AI language model, I do not",
    "I know as an AI language model you don't have",
    "as an AI language model, you cannot",
    "I'm sorry, but as an AI language model",
    "As an AI language model, I don't have",
    "I'm an AI ",
    "I am an AI ",
    "As your dedicated AI language model",
    "As a hypothetical AI",
    "As a neutral AI",
    "my knowledge cutoff",
    "my knowledge cut off",
    "As a machine",
    "I cannot assist",
    "I do not have personal preferences",
    "I don't have personal preferences",
    "Unfortunately, I cannot provide",
    "I'm sorry, I cannot",
    "I'm sorry, I cannot generate",
    "AI cannot create or program",
    "I'm afraid I cannot create",
    "OpenAI",
]


def build_sft_llama_datasets(data_path):
    tokenizer = get_tokenizer()
    args = get_args()
    seq_length = args.seq_length

    data_files = {"train": os.path.join(data_path)}
    extension = "json"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir="",
        use_auth_token=False,
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names_train = raw_datasets["train"].column_names
    train_dataset = raw_datasets["train"]

    def print_supervised_dataset_example(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    def preprocess_data(dataset):
        model_inputs = {
            "input_ids": [],
            "labels": [],
            "loss_mask": [],
            "ignore_lengths": []
        }
        max_seq_length = seq_length
        max_source_length = seq_length // 2
        max_target_length = seq_length // 2
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        for i in range(len(dataset['instruction'])):
            input_data = dataset['input'][i].strip()
            output_data = dataset['output'][i].strip()
            instruct_data = dataset['instruction'][i].strip()
            row = {
                "input": input_data,
                "output": output_data,
                "instruction": instruct_data
            }

            if len(row["input"]) > 0:
                input_ = prompt_input.format_map(row)
            else:
                input_ = prompt_no_input.format_map(row)
            if (_filter_by_words(input_) is None) or (_filter_by_words(row["output"]) is None):
                continue
            output_ = f"{row['output']}"
            a_ids = tokenizer.encode(text=input_, add_special_tokens=False)
            b_ids = tokenizer.encode(text=output_, add_special_tokens=False)
            if len(a_ids) > max_source_length:
                a_ids = a_ids[:max_source_length]
            if len(b_ids) > max_target_length - 1:
                b_ids = b_ids[:max_target_length - 1]
            input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
            labels = [IGNORE_INDEX] * len(a_ids) + b_ids + [tokenizer.eos_token_id]
            ignore_idx = [len(a_ids) - 1]
            pad_len = max_seq_length - len(input_ids)
            if pad_len > 0:
                ignore_idx.append(len(input_ids) - 2)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
            labels = [(l if l != tokenizer.pad_token_id else IGNORE_INDEX) for l in labels]
            loss_mask = [(0 if k == IGNORE_INDEX or k == tokenizer.eos_token_id or k == tokenizer.bos_token_id else 1)
                         for k in labels]

            model_inputs["loss_mask"].append(loss_mask)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
            model_inputs["ignore_lengths"].append(ignore_idx)

        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_data,
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


def _filter_by_words(text: str, filter_words: list[str] | None = None) -> None | str:
    """Used to filter text that contains one of the `FILTER_BY_WORDS`. If so we return `None`
       otherwise we return the string

    Args:
        text (str): text to be filtered

    Returns:
        None | str: filtered text
    """
    filter_words = filter_words or FILTER_BY_WORDS
    for word in filter_words:
        if word.lower() in text.lower():
            return None
    return text
