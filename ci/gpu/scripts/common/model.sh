#!/bin/bash
set -eo pipefail

root_path=$(dirname "$(readlink -f "$0")")
export scripts_root_path=${root_path}/..
source ${scripts_root_path}/common/common.sh

# 所存储的日志文件目录
agile_model_file_log_path=${model_file_log_path:-"/workspace/logs"}
mkdir -p ${agile_model_file_log_path}

# 检查训练任务包含"lm loss:" 字符串的数量
training_lm_loss_times=5
# 检查最长时间
timeout=${timeout:-1800}

function check_training_finish() {
    local log_file_dir=$1

    # 计数器和最长等待时间（以秒为单位）
    counter=0
    lm_loss_counter=0

    # 初始化最后一行变量
    last_line_log=""

    # 循环检查文件内容
    while [ $counter -lt $timeout ]; do
        if [[ $lm_loss_counter -ge $training_lm_loss_times ]]; then
            SUCCESS_echo_date "模型训练完成"
            break
        fi

        if grep -q "Root Cause" "$log_file_dir"; then
            FAILURE_echo_date "模型训练启动失败, 请检查日志:"
            cat $log_file_dir
            exit -1
        fi

        local latest_log=$(tail -n 1 $log_file_dir)
        if grep -q "lm loss:" "$log_file_dir"; then
            # 查看当前日志中出现 "lm loss:" 的次数
            local current_lm_loss_times=$(grep -c "lm loss:" $log_file_dir)
            local lm_loss_counter=$current_lm_loss_times
            SUCCESS_echo_date "训练任务中包含 'lm loss:' 第 ${lm_loss_counter}/$training_lm_loss_times 次, 最新[1]行日志: $latest_log"
        else
            # 检查新行是否和最后一行不同
            if [ "$latest_log" != "$last_line_log" ]; then
                # 如果不同，打印新行并更新最后一行
                SUCCESS_echo_date "启动中, 最新[1]行日志: $latest_log"
                last_line_log=$latest_log
            fi
        fi
        
        # 休眠 1 秒
        sleep 1

        # 更新计数器
        counter=$((counter + 1))
    done

    # 检查计数器是否超过了最长等待时间
    if [ $counter -eq $timeout ]; then
        FAILURE_echo_date "超过最长等待时间, 模型训练任务没有正常启动或完成, 查看日志: $log_file_dir"
        cat $log_file_dir
        exit 1
    fi
}

function stop_running_training() {
    if ps -ef | grep 'python -m torch.distributed.launch' | grep -v grep > /dev/null; then
        pkill -f 'python -m torch.distributed.launch'
    fi

    sleep 10
}

function run_models_case() {
    pipe_name="$1"
    models_name_list="$2"
    node_nums=$3
    GPU_COUNT=$4

    # RANK=0 代表master， RANK 非0 代表worker
    local role=master
    if [[ ! $RANK -eq 0 ]]; then
        role=worker
    fi

    SUCCESS_echo_date "Begin run pipe ${pipe_name}."
    field_path=${scripts_root_path}/testcase/${pipe_name}

    case_id=1
    total_cases=$(echo $models_name_list | wc -w)

    for model_name in ${models_name_list}
    do
        starttime=`date +%s%N`
        SUCCESS_echo_date "Run ${model_name} in ${pipe_name}, ${case_id} / ${total_cases}."

        if [[ $node_nums -eq 1 ]]; then
            # 停止训练任务
            stop_running_training
        fi

        log_file_dir=${agile_model_file_log_path}/${model_name}_${node_nums}_${role}_training.log
        # 放到后台训练
        nohup bash ${field_path}/run.sh ${model_name} $GPU_COUNT > $log_file_dir 2>&1 &

        # 检查是否符合预期
        check_training_finish $log_file_dir

        # 收集该模型性能指标
        # collect_perf_metric $model_name $log_file_dir

        endtime=`date +%s%N`
        c=`expr $endtime - $starttime`
        c=`expr $c / 1000000`
        SUCCESS_echo_date "Finish test ${model_name} in ${pipe_name}, ${case_id} / ${total_cases}. Time cost: $c ms."
        case_id=$((case_id + 1))
    done

    SUCCESS_echo_date "Finish run pipe ${pipe_name}."

    # 停留几秒，给后置收集性能数据动作留出buffer
    sleep 5

}

run_models_case "$@"