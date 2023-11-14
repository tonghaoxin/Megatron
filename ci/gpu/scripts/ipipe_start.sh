#!/bin/bash
set -eo pipefail

root_path=$(dirname "$(readlink -f "$0")")
export scripts_root_path=${root_path}
source ${scripts_root_path}/common/common.sh

case_type="test_function"

# 如若使用其他机器使用，则需要配置kubeconfig和kubectl、kubectl_view_allocations_path目录地址
kubectl_path="/usr/local/bin"
kubectl_view_allocations_path="/usr/local/bin"

# "/ssd2/leoli/kubeconfigs/configs/hwl-cce.config"、"/ssd2/leoli/kubeconfigs/configs/bjtest_ztl-ai-1.24.config"
kubeconfig_path=${kubeconfig_path:-"/ssd2/leoli/kubeconfigs/configs/hwl-cce.config"}
# "baidu.com/a800_80g_cgpu"、"baidu.com/a100_80g_cgpu"
gpu_resource=${gpu_resource:-"baidu.com/a800_80g_cgpu"}
gpu_count="8"

export PATH=$PATH:${kubectl_path}:${kubectl_view_allocations_path}
export KUBECONFIG="$kubeconfig_path"
export GPU_RESOURCE="$gpu_resource"
export GPU_COUNT="$gpu_count"

# K8S 集群方式存在于pfs 数据的根目录
export TRAIN_DATA_DIR="/mnt/pfs/leoli"
export IMAGE=${IMAGE:-"registry.baidubce.com/cce-ai-native/aiak-megatron:ubuntu22.04-cu12.2-torch2.1.0-py310_20231114_111918"}
# export SPECIFIC_PYTORCHJOB_COMMAND="sleep 1d" # 可用于单机调试
export TIMEOUT=${TIMEOUT:-"3600"}
export SCHEDULE_TIMEOUT=${SCHEDULE_TIMEOUT:-"300"}
# export specific_model_name="galactica-30b" # 运行指定模型

export BOS_SYNC_MEGATRON_SCRIPTS_ADDR=${BOS_SYNC_MEGATRON_SCRIPTS_ADDR:-"https://cce-ai-datasets.bj.bcebos.com/hac_test/megatron/project_compile/20231114-001/script.tar.gz"}

function usage()
{
	echo -e "bash $0 [args]"
	echo -e ""
	echo -e "\t--case_type: test_function | test_perf"
  echo -e ""
  echo -e "\tExample: bash ipipe_start.sh --case_type=test_function"
	echo -e ""
}

while [ $# -gt 0 ]; do
  case "$1" in
    --case_type=*)
      case_type="${1#*=}"
      ;;
    *)
      printf "*********************************************\n"
      printf "Error: Invalid argument. $1\n"
      usage
      printf "*********************************************\n"
      exit 1
  esac
  shift
done

# 功能测试/性能测试: test_function | test_perf
run_all_ipipe_case "${case_type}"
