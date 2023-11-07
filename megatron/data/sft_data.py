#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
"""SFT data"""

from typing import List, Literal, Optional, Any, Dict, Generator, Tuple
from dataclasses import dataclass, field
from megatron import get_args, get_tokenizer
from megatron.data.template import get_template_and_fix_tokenizer
from transformers import HfArgumentParser
import sys
import os
import json


@dataclass
class DatasetAttr:
    """dataset attr"""
    load_from: str
    dataset_name: Optional[str] = None
    dataset_sha1: Optional[str] = None
    system_prompt: Optional[str] = None

    def __repr__(self) -> str:
        """__repr__"""
        return self.dataset_name

    def __post_init__(self):
        """__post_init__"""
        self.prompt = "instruction"
        self.query = "input"
        self.response = "output"
        self.history = None


@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    sft_data_template: str = field(
        metadata={"help": "Which template to use for constructing prompts in training and inference."}
    )
    sft_dataset_name: Optional[str] = field(
        default="alpaca_en",
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="/workspace/dataset",
        metadata={"help": "The name of the folder containing datasets."}
    )
    split: Optional[str] = field(
        default="train",
        metadata={"help": "Which dataset split to use for training and evaluation."}
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable streaming mode."}
    )
    buffer_size: Optional[int] = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in streaming mode."}
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing."}
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum total output sequence length after tokenization."}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."}
    )
    eval_num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"}
    )
    ignore_pad_token_for_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "System prompt to add before the user query. "
                          "Use `|` to separate multiple prompts in training."}
    )
    val_size: Optional[float] = field(
        default=0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."}
    )


def _parse_args(parser: HfArgumentParser, args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    """_parse_args"""
    if args is not None:
        return parser.parse_dict(args, True)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()


def parse_data_args(
    args: Optional[Dict[str, Any]] = None
) -> Tuple[DataArguments]:
    """parse data args"""
    parser = HfArgumentParser((
        DataArguments,
    ))
    return _parse_args(parser, args)


def init_for_sft(dataset_mode: str, data_args: DataArguments) -> DatasetAttr:
    """init for sft"""
    dataset_name = data_args.sft_dataset_name
    dataset_info = dict()
    with open(os.path.join(data_args.dataset_dir, "dataset_info.json"), "r") as f:
        dataset_info = json.load(f)

    if dataset_mode == "hf_hub_url":
        dataset_attr = DatasetAttr("hf_hub", dataset_name="train")
    elif dataset_mode == "script_url":
        dataset_attr = DatasetAttr("script", dataset_name="train")
    else:
        dataset_attr = DatasetAttr(
            "file",
            dataset_name=dataset_name,
        )

    if "columns" in dataset_info[dataset_name]:
        dataset_attr.prompt = dataset_info[dataset_name]["columns"].get("prompt", None)
        dataset_attr.query = dataset_info[dataset_name]["columns"].get("query", None)
        dataset_attr.response = dataset_info[dataset_name]["columns"].get("response", None)
        dataset_attr.history = dataset_info[dataset_name]["columns"].get("history", None)

    return dataset_attr


def construct_example(examples: Dict[str, List[Any]]) -> Generator[Any, None, None]:
    """construct example"""
    for i in range(len(examples["prompt"])):
        query, response = examples["prompt"][i], examples["response"][i]
        query = query + "\n" + examples["query"][i] if "query" in examples and examples["query"][i] else query
        history = examples["history"][i] if "history" in examples else None
        system = examples["system"][i] if "system" in examples else None
        yield query, response, history, system


def preprocess_supervised_dataset(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
    """preprocess supervised dataset"""
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    ignore_index = -100
    model_inputs = {"input_ids": [], "loss_mask": [], "labels": []}
    args = get_args()
    tokenizer = get_tokenizer()
    max_length = args.seq_length
    max_source_length = max_length // 2
    max_target_length = max_length // 2

    template_name = args.sft_data_template
    template = get_template_and_fix_tokenizer(template_name, tokenizer)

    for query, response, history, system in construct_example(examples):
        input_ids, labels, loss_masks = [], [], []

        for source_ids, target_ids in template.encode_multiturn(tokenizer, query, response, history, system):
            if len(source_ids) > max_source_length:
                source_ids = source_ids[:max_source_length]
            if len(target_ids) > max_target_length - 1:
                target_ids = target_ids[:max_target_length - 1] + [tokenizer.eos_token_id]

            if len(input_ids) + len(source_ids) + len(target_ids) > max_length:
                break
            if all(v == tokenizer.eos_token_id for v in target_ids):
                continue

            input_id = source_ids + target_ids
            input_ids += input_id
            label = [ignore_index] * len(source_ids) + target_ids
            labels += label
            loss_masks += [(0 if k == ignore_index or k == tokenizer.eos_token_id else 1)
                          for k in label]

        if len(input_ids) == 0:
            continue
        if not args.variable_seq_lengths:
            padding_side = tokenizer.padding_side
            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                if padding_side == "right":
                    input_ids += [tokenizer.pad_token_id] * pad_len
                    labels += [ignore_index] * pad_len
                    loss_masks += [0] * pad_len
                elif padding_side == "left":
                    input_ids = [tokenizer.pad_token_id] * pad_len + input_ids
                    labels = [ignore_index] * pad_len + labels
                    loss_masks = [0] * pad_len + loss_masks

        model_inputs["input_ids"].append(input_ids)
        model_inputs["loss_mask"].append(loss_masks)
        model_inputs["labels"].append(labels)

    return model_inputs
