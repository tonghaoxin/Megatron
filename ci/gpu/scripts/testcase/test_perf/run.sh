#!/bin/bash
set -eo pipefail

function run() {
    source ${scripts_root_path}/common/common.sh
    local model_name=$1
    local GPU_COUNT=${2:-"8"}
    local model_start_path=${scripts_root_path}/models_parameters/${model_name}.sh

    sed -i "/GPUS_PER_NODE=/s/^\(\s*\).*/\1\GPUS_PER_NODE=$GPU_COUNT/" $model_start_path

    SUCCESS_echo_date "开始训练模型: $model_name"
    bash $model_start_path
    SUCCESS_echo_date "结束训练模型: $model_name"
}
run $@