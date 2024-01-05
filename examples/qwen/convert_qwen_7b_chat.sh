#! /bin/bash

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

python $ROOT_DIR/../../tools/convert_checkpoint/convert_qwen_hf_megatron.py \
--load_path /ssd2/models/Qwen/Qwen-7B-Chat-HF/ \
--save_path /ssd2/models/Qwen/Qwen-7B-Chat/ \
--target_params_dtype bf16 \
--megatron-path /workspace/Megatron-LM/ \
--target_tensor_model_parallel_size 1 \
--target_pipeline_model_parallel_size 1 \
--target_data_parallel_size 8 \
--use_distributed_optimizer \
--model_name qwen-7b \
#--print-checkpoint-structure \
