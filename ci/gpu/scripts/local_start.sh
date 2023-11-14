#!/bin/bash
set -eo pipefail

root_path=$(dirname "$(readlink -f "$0")")
export scripts_root_path=${root_path}

# K8S 集群方式存在于pfs 数据的根目录
export TRAIN_DATA_DIR="/mnt/pfs/leoli"

pipe_name="test_function"
model_name="llama-7b"

# 方法1: 直接使用原始脚本训练
model_run_path=${scripts_root_path}/testcase/${pipe_name}/run.sh
bash $model_run_path $model_name

# 方法2: 使用增加了校验/判断/日志简化等功能的封装脚本
# model_run_path=${scripts_root_path}/common/model.sh
# bash $model_run_path $pipe_name $model_name