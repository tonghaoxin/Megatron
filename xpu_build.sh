#!/bin/bash
set -euo pipefail
echo "========= xpu build enter ========="
if [[ -n $XPU_MEGATRON_OUTPUT ]]; then
    mkdir -p output
    pushd output
    wget -O megatron-0.1.tar.gz $XPU_MEGATRON_OUTPUT
    tar -zxvf megatron-0.1.tar.gz                                                                                                                                                 
    rm  megatron-0.1.tar.gz
    popd
 fi