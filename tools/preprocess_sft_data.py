#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing data for SFT."""

import argparse
import json
import multiprocessing
import os
import sys
import time
from typing import List, Any, Dict
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from megatron.data.template import get_template_and_fix_tokenizer


class Encoder(object):
    def __init__(self, args):
        """__init__"""
        self.args = args
        Encoder.tokenizer = build_tokenizer(self.args)
        self.template = get_template_and_fix_tokenizer(self.args.template_name, Encoder.tokenizer)

    def initializer(self):
        """initializer"""
        # Use Encoder class as a container for global data
        print(f"initializer {self.args}")

    def _single_turn_template(self, data: Dict[str, List[Any]]):
        """_single_turn_template"""
        prompt, query, response, history = data["instruction"], data["input"], data["output"], None
        if len(query) > 0:
            query = prompt + "\n" + query
        else:
            query = prompt
        system = data["system"] if "system" in data else None
        return query, response, history, system

    def _multi_turn_template(self, data: Dict[str, List[Any]], is_oaast: bool):
        """_multi_turn_template"""
        query, response, history, system = None, None, None, None
        if not is_oaast:
            conversations = data["conversations"]
            history_conversations = conversations[:-2]
            history = []
            for i, conversation in enumerate(history_conversations):
                index = i // 2
                if i % 2 == 0:
                    if "prompt" in conversation:
                        history.append([conversation["prompt"] + "\n" + conversation["value"]])
                    else:
                        history.append([conversation["value"]])
                else:
                    history[index].append(conversation["value"])
            query, response = conversations[-2]["value"], conversations[-1]["value"]
            query = conversations[-2]["prompt"] + "\n" + query if "prompt" in conversations[-2] else query
            system = data["system"] if "system" in data else None
        else:
            prompt, query, response, history = data["instruction"], data["input"], data["output"], data["history"]
            if len(query) > 0:
                query = prompt + "\n" + query
            else:
                query = prompt
            system = data["system"] if "system" in data else None
        return query, response, history, system

    def construct_template(self, data: Dict[str, List[Any]], is_multi_turn: bool, is_oaast: bool):
        """construct_template"""
        if is_multi_turn:
            return self._multi_turn_template(data, is_oaast)
        return self._single_turn_template(data)

    def encode(self, json_line):
        """encode"""
        data = json.loads(json_line)
        ids = {}
        max_length = self.args.seq_length
        max_source_length = max_length // 2
        max_target_length = max_length // 2
        is_multi_turn = False
        if self.args.template_type == "multiTurn":
            is_multi_turn = True
        is_oaast = self.args.oaast_dataset_type
        query, response, history, system = self.construct_template(data, is_multi_turn, is_oaast)
        input_ids, src_tgt_lens = [], []
        for source_ids, target_ids in self.template.encode_multiturn(Encoder.tokenizer, query, response, history,
                                                                     system):
            if len(source_ids) > max_source_length:
                source_ids = source_ids[:max_source_length]
            if len(target_ids) > max_target_length - 1:
                target_ids = target_ids[:max_target_length - 1] + [Encoder.tokenizer.eos_token_id]

            if len(input_ids) + len(source_ids) + len(target_ids) > max_length:
                break
            if all(v == Encoder.tokenizer.eos_token_id for v in target_ids):
                continue

            input_id = source_ids + target_ids
            input_ids += input_id
            src_tgt_lens += [len(source_ids)] + [len(target_ids)]
        ids["input_ids"] = input_ids
        input_lens = len(input_ids)
        src_tgt_size = len(src_tgt_lens)
        if input_lens - src_tgt_size >= 2:
            src_tgt_lens = [-1] + src_tgt_lens + [-1]
            src_tgt_lens += [0] * (input_lens - src_tgt_size - 2)
        else:
            raise ValueError(
                f"invalid input_ids: {input_ids}")
        ids["src_tgt_lens"] = src_tgt_lens

        return ids, len(json_line)


def get_args():
    """get_args"""
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--template-type', type=str, required=True,
                       choices=['multiTurn', 'singleTurn'],
                       help='parse the jsonl for sft template data')
    group.add_argument('--template-name', type=str, required=True,
                       help='template name for dataset')
    group.add_argument('--oaast-dataset-type', type=bool, default=False, required=False,
                       help='Use oaast dataset template to parse')
    group.add_argument('--seq-length', type=int, default=2048,
                       help='Maximum sequence length to process.')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase', 'BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'NullTokenizer',
                                'LLaMASentencePieceTokenizer', 'GalacticaHFTokenizer',
                                'ChatGLMTokenizer', 'IceTokenizer', 'OptTokenizer',
                                'BloomHFTokenizer', 'LLaMASFTTokenizerHF'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='sentencepeice tokenizer model.')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--hf-tokenizer-path', type=str, default=None,
                       help='HuggingFace tokenizer path: '
                            '1) A string, the *model id* of a predefined tokenizer '
                            'hosted inside a model repo on huggingface.co'
                            '2) A path to a *directory* containing vocabulary '
                            'files required by the tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk-size', type=int, required=True,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    args.pad_vocab_size_to = None

    return args


def main():
    """main"""
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_tos = pool.imap(encoder.encode, fin, args.chunk_size)

    level = "tokens"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = dict()
    output_bin_files["input_ids"] = "{}_input_ids_{}.bin".format(args.output_prefix, level)
    output_idx_files["input_ids"] = "{}_input_ids_{}.idx".format(args.output_prefix, level)
    builders["input_ids"] = indexed_dataset.make_builder(output_bin_files["input_ids"],
                                                         impl=args.dataset_impl,
                                                         vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    for i, (tos, bytes_processed) in enumerate(encoded_tos, start=1):
        total_bytes_processed += bytes_processed
        input_data = []
        for key, tokens in tos.items():
            if len(tokens) == 0:
                continue
            input_data.append(tokens)
        if len(input_data) != 2:
            continue
        builders["input_ids"].add_item(torch.IntTensor(input_data))
        builders["input_ids"].end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} datasets",
                  f"({i / elapsed} datasets/s, {mbs} MB/s).",
                  file=sys.stderr)
    print("Done! Now finalizing.")

    builders["input_ids"].finalize(output_idx_files["input_ids"])


if __name__ == '__main__':
    main()
