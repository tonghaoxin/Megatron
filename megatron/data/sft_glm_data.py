# !/usr/bin/env python3
"""SFT GLM data"""
from __future__ import annotations
from megatron import get_tokenizer, get_args, print_rank_0
from datasets import load_dataset
import os
from typing import List, Optional, Dict
import numpy as np
import torch
import copy
import json
import pickle


def build_glm_sft_datasets(data_path):
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
        print("text:\n{}".format(example["text"]))
        print("target:\n{}".format(example["target"]))
        print("attention_mask:\n{}".format(example["attention_mask"]))
        print("loss_mask:\n{}".format(example["loss_mask"]))
        print("position_id:\n{}".format(example["position_id"]))

    def preprocess_data(dataset):
        model_inputs = {
            "text": [],
            "target": [],
            "attention_mask": [],
            "loss_mask": [],
            "position_id": [],
        }
        max_seq_length = seq_length
        max_source_length = max_seq_length // 2
        max_target_length = max_seq_length // 2
        cls_id = tokenizer.get_command('ENC').Id
        mask_token = 'sMASK' if args.task_mask else 'MASK'
        mask_id = tokenizer.get_command(mask_token).Id
        eos_id = tokenizer.get_command('eos').Id
        pad_id = tokenizer.get_command('pad').Id
        sop_id = tokenizer.get_command('sop').Id
        eop_id = tokenizer.get_command('eop').Id
        for i in range(len(dataset['instruction'])):
            row = {
                "input": dataset['input'][i] if len(dataset['input']) > 0 else "",
                "output": dataset['output'][i],
                "instruction": dataset['instruction'][i]
            }
            if len(row["input"]) > 0:
                input_ = row["input"]
            else:
                input_ = row["instruction"]
            output_ = row["output"]
            source_tokens = tokenizer.EncodeAsIds(input_).tokenization
            if len(source_tokens) + 3 > max_source_length:
                source_tokens = source_tokens[-(max_source_length - 3):]
            source_tokens = [cls_id] + source_tokens + [mask_id, eos_id]
            context_length = len(source_tokens)
            if len(source_tokens) < max_source_length:
                source_tokens = source_tokens + [pad_id] * (max_source_length - len(source_tokens))
            sep = len(source_tokens)
            position_ids = list(range(len(source_tokens)))
            block_position_ids = [0] * len(source_tokens)
            mask_pos = source_tokens.index(mask_id)
            target_tokens = tokenizer.EncodeAsIds(" " + output_).tokenization
            target_tokens = target_tokens + [eop_id]
            if len(target_tokens) > max_target_length:
                target_tokens = target_tokens[:max_target_length]
            loss_mask = [1] * len(target_tokens)
            if len(target_tokens) < max_target_length:
                loss_mask += [0] * (max_target_length - len(target_tokens))
                target_tokens += [pad_id] * (max_target_length - len(target_tokens))
            if args.mask_pad_token:
                attention_mask = np.zeros((max_source_length + max_target_length,
                                           max_source_length + max_target_length), dtype=np.int64)
                attention_mask[:, :context_length] = 1
                attention_mask[max_source_length:, max_source_length:] = np.tril(
                    np.ones((max_target_length, max_target_length), dtype=np.int64))
                attention_mask = attention_mask[None, :, :]
            else:
                attention_mask = np.array(sep, dtype=np.int64)
            tokens = source_tokens + [sop_id] + target_tokens[:-1]
            loss_mask = [0] * len(source_tokens) + loss_mask
            target_ids = [0] * len(source_tokens) + target_tokens
            position_ids += [mask_pos] * len(target_tokens)
            if args.no_block_position:
                block_position_ids += [1] * len(target_tokens)
            else:
                block_position_ids += list(range(1, len(target_tokens) + 1))
            position_ids = [position_ids, block_position_ids]
            text = np.array(tokens, dtype=np.int64)
            target = np.array(target_ids, dtype=np.int64)
            model_inputs["text"].append(text)
            model_inputs["target"].append(target)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["loss_mask"].append(np.array(loss_mask, dtype=np.int64))
            model_inputs["position_id"].append(np.array(position_ids, dtype=np.int64))
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_data,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names_train,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset",
    )
    print_supervised_dataset_example(train_dataset[0])

    train_dataset.set_format(
        type=train_dataset.format["type"], columns=train_dataset.column_names,
        format_kwargs=train_dataset.format["format_kwargs"]
    )

    return train_dataset


class CustomizationDataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, tokenizer):
        self.max_src_length, self.max_tgt_length = args.src_seq_length, args.tgt_seq_length
        self.tokenizer = tokenizer
        self.mask_pad_token = args.mask_pad_token
        self.no_block_position = args.no_block_position
        self.task_mask = args.task_mask
        self.dataset_name = "train"
        print_rank_0(f"Creating dataset from {data_path}")
        source_texts, target_texts = [], []
        with open(os.path.join(f"{data_path}"),
                  encoding='utf-8') as file:
            data = json.load(file)
            for row in data:
                source = row['input'] if len(row['input']) > 0 else row['instruction']
                source_texts.append(source)
                target = row['output']
                target_texts.append(target)
        self.examples, self.example_list = {}, []
        for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
            if (idx + 1) % 20000 == 0:
                print_rank_0(f"Complete {idx + 1} examples")
            guid = "%s-%s" % ("train", idx)
            meta = {"ref": target_text}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            self.examples[guid] = example
            self.example_list.append(example)
        print_rank_0(f"Return {len(self.examples)} train examples")

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        example = self.example_list[idx]
        cls_id = self.tokenizer.get_command('ENC').Id
        mask_token = 'sMASK' if self.task_mask else 'MASK'
        mask_id = self.tokenizer.get_command(mask_token).Id
        eos_id = self.tokenizer.get_command('eos').Id
        pad_id = self.tokenizer.get_command('pad').Id
        sop_id = self.tokenizer.get_command('sop').Id
        eop_id = self.tokenizer.get_command('eop').Id
        source_text, target_text = example.text_a, example.text_b
        source_tokens = self.tokenizer.EncodeAsIds(source_text).tokenization
        if len(source_tokens) + 3 > self.max_src_length:
            source_tokens = source_tokens[-(self.max_src_length - 3):]
        source_tokens = [cls_id] + source_tokens + [mask_id, eos_id]
        context_length = len(source_tokens)
        if len(source_tokens) < self.max_src_length:
            source_tokens = source_tokens + [pad_id] * (self.max_src_length - len(source_tokens))
        sep = len(source_tokens)
        position_ids = list(range(len(source_tokens)))
        block_position_ids = [0] * len(source_tokens)
        mask_pos = source_tokens.index(mask_id)

        target_tokens = self.tokenizer.EncodeAsIds(" " + target_text).tokenization
        target_tokens = target_tokens + [eop_id]
        if len(target_tokens) > self.max_tgt_length:
            target_tokens = target_tokens[:self.max_tgt_length]
        loss_mask = [1] * len(target_tokens)
        if len(target_tokens) < self.max_tgt_length:
            loss_mask += [0] * (self.max_tgt_length - len(target_tokens))
            target_tokens += [pad_id] * (self.max_tgt_length - len(target_tokens))
        if self.mask_pad_token:
            attention_mask = np.zeros((self.max_src_length + self.max_tgt_length,
                                       self.max_src_length + self.max_tgt_length), dtype=np.int64)
            attention_mask[:, :context_length] = 1
            attention_mask[self.max_src_length:, self.max_src_length:] = np.tril(
                np.ones((self.max_tgt_length, self.max_tgt_length), dtype=np.int64))
            attention_mask = attention_mask[None, :, :]
        else:
            attention_mask = np.array(sep, dtype=np.int64)
        tokens = source_tokens + [sop_id] + target_tokens[:-1]
        loss_mask = [0] * len(source_tokens) + loss_mask
        target_ids = [0] * len(source_tokens) + target_tokens
        position_ids += [mask_pos] * len(target_tokens)
        if self.no_block_position:
            block_position_ids += [1] * len(target_tokens)
        else:
            block_position_ids += list(range(1, len(target_tokens) + 1))
        position_ids = [position_ids, block_position_ids]
        sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                  'attention_mask': attention_mask,
                  'loss_mask': np.array(loss_mask, dtype=np.int64),
                  "position_id": np.array(position_ids, dtype=np.int64), "uid": example.guid}
        return sample


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta: Optional[Dict] = None, idx=-1,
                 num_choices=1):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.num_choices = num_choices
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)