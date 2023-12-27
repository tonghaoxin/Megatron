#!/bin/bash
set -x
while true; do
    xpu_smi > xpu_log
    sed -i -e '1,/PROCESSES/d' xpu_log
    grep -E "^\|[[:space:]]+[0-9]" xpu_log | awk -F "|" '{print $3}' > process_log
    if grep -r "[1-9]" process_log; then
        echo "cards is in use!"
        sleep 120
    else
        xpu_smi -m | awk '{print $16 $18 $20}'> memory_log
        if grep -r "[1-9]" memory_log; then
            sleep 120
        else
            echo "No card in use!"
            bash change_kernel_time.sh 0,1,2,3,4,5,6,7 3600000
            exit 0
        fi
    fi
done
