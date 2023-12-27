#!/usr/bin/env bash

eval `echo $1 | awk -F, '{for(i=1;i<=NF;++i) printf("devices[%d]=%d;",((i-1)),$i);}' `
timeout=$2
for device in ${devices[@]}
do
    echo "whosyourdaddy" >> /proc/xpu/dev${device}/info
    current_threshold=`cat /proc/xpu/dev${device}/task_timeout_detect_threshold_in_ms`
    echo "Before dev:${device} threshold:${current_threshold}"
    echo ${timeout} > /proc/xpu/dev${device}/task_timeout_detect_threshold_in_ms
    current_threshold=`cat /proc/xpu/dev${device}/task_timeout_detect_threshold_in_ms`
    echo "After dev:${device} threshold:${current_threshold}"
done
