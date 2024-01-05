#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

python $ROOT_DIR/../../tools/preprocess_data_qwen.py \
    --input /ssd2/datasets/pile_test/test.jsonl \
    --json-keys content \
    --append-eod \
    --split-sentences \
    --tokenizer-type QwenTokenizer \
    --tokenizer-model /ssd2/models/Qwen/Qwen-7B/qwen.tiktoken \
    --output-prefix /home/datasets/qwen_test/qwen_00_text_document \
    --workers 16 \
    --chunk-size 25 \
    --seq-length 2048
