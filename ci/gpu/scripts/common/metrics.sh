#!/bin/bash
set -eo pipefail

root_path=$(dirname "$(readlink -f "$0")")
export scripts_root_path=${root_path}/..
source ${scripts_root_path}/common/common.sh

function summary_perf_metrics() {
    # 将性能指标数据从容器中cp出来, 日志文件在容器 "/workspace/logs" 目录下

    metric_file=${scripts_root_path}/total_metric.csv
    echo "model_name,TP_PP_DP_ZERO,micro_batch_size,global_batch_size,seq_length,elapsed_time_per_iteration,lr,lm_loss,吞吐(B/台/天)" > $metric_file

    collection_model_flag=""

    model_training_logs=`ls ${scripts_root_path}/logs |grep training.log`
    for model_training_log in ${model_training_logs};
    do
        # training log name 例子: galactica-30b_2_worker_training.log        
        local model_name=`echo $model_training_log | awk -F'_' '{print $(1)}'`
        local RANK=`echo $model_training_log | awk -F'_' '{print $(2)}'`
        local role=`echo $model_training_log | awk -F'_' '{print $(3)}'`

        if  [[ "${collection_model_flag}" =~ "$model_name" ]];then
            continue
        fi

        collection_model_flag="${model_name},${collection_model_flag}"

        # RANK=0 代表master， RANK 非0 代表worker
        master_log_name="${model_name}_${RANK}_master_training.log"
        model_training_log_args_path="${scripts_root_path}/logs/${master_log_name}"
        metrics_model_training_log_path=${scripts_root_path}/logs/${master_log_name}
        if [[ "${RANK}" != "1" ]];then
            worker_log_name="${model_name}_${RANK}_worker_training.log"
            metrics_model_training_log_path=${scripts_root_path}/logs/${worker_log_name}
        fi

        args_file_path="${scripts_root_path}/logs/${model_name}_args.txt"

        sed -n "/------------------------ arguments ------------------------/,/end of arguments/p" ${model_training_log_args_path} > ${args_file_path}
        tensor_model_parallel_size=`grep '  tensor_model_parallel_size' $args_file_path | awk '{print $NF}'`
        pipeline_model_parallel_size=`grep '  pipeline_model_parallel_size' $args_file_path | awk '{print $NF}'`
        data_parallel_size=`grep '  data_parallel_size' $args_file_path | awk '{print $NF}'`
        micro_batch_size=`grep '  micro_batch_size' $args_file_path | awk '{print $NF}'`
        global_batch_size=`grep '  global_batch_size' $args_file_path | awk '{print $NF}'`
        seq_length=`grep '  seq_length' $args_file_path | awk '{print $NF}'`
        use_distributed_optimizer=`grep '  use_distributed_optimizer' $args_file_path | awk '{print $NF}'`
        lr=`grep '  lr ' $args_file_path | awk '{print $NF}'`

        elapsed_time_per_iteration=`grep "elapsed time per iteration (ms):" $metrics_model_training_log_path |grep 'elapsed time per iteration (ms)' | tail -n 1 | awk 'BEGIN{FS="elapsed time per iteration \\\(ms\\\):";OFS=" "} {print $2}' |awk -F '|' '{print $1}'|awk '{print $1}'`
        lm_loss=`grep "elapsed time per iteration (ms):" $metrics_model_training_log_path |grep 'lm loss' | tail -n 1  | awk 'BEGIN{FS="lm loss:";OFS=" "} {print $2}' |awk -F '|' '{print $1}'|awk '{print $1}'`

        # 吞吐
        # 则性能折合每台GPU机器每天的Tokens处理吞吐量（Billion，10亿）计算为：
        # = global batch size * seq-length / elapsed time per iteration
        # = 128*1024/(3920.6/1000)*86400/1000000000=2.9（B/台/天）
        through=`echo "$global_batch_size*$seq_length/($elapsed_time_per_iteration/1000)*86400/1000000000" | bc -l | awk '{printf "%.2f\n", $0}'`

        SUCCESS_echo_date "------------------------ 性能指标收集 start ------------------------"
        SUCCESS_echo_date "模型: $model_name"
        SUCCESS_echo_date "pipeline_model_parallel_size: $pipeline_model_parallel_size"
        SUCCESS_echo_date "tensor_model_parallel_size: $tensor_model_parallel_size"
        SUCCESS_echo_date "data_parallel_size: $data_parallel_size"
        SUCCESS_echo_date "micro_batch_size: $micro_batch_size"
        SUCCESS_echo_date "global_batch_size: $global_batch_size"
        SUCCESS_echo_date "seq_length: $seq_length"
        SUCCESS_echo_date "elapsed_time_per_iteration: $elapsed_time_per_iteration"
        SUCCESS_echo_date "lr: $lr"
        SUCCESS_echo_date "lm_loss: $lm_loss"
        SUCCESS_echo_date "through: $global_batch_size*$seq_length/($elapsed_time_per_iteration/1000)*86400/1000000000=$through"

        echo "$model_name,${tensor_model_parallel_size}_${pipeline_model_parallel_size}_${data_parallel_size}_${use_distributed_optimizer},$micro_batch_size,$global_batch_size,$seq_length,$elapsed_time_per_iteration,\"\"\"${lr}\"\"\",\"\"\"$lm_loss\"\"\",$through" >> $metric_file
        SUCCESS_echo_date "------------------------ 性能指标收集 end ------------------------"
    done

}

# 汇总性能数据
summary_perf_metrics "$@"