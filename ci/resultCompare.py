# !/usr/bin/env python3

"""compare loss"""
import os
import json
import argparse
import subprocess

base_dir = os.path.abspath(os.path.join(os.getcwd(), "ci/baseline.json"))

def get_args():
    """获取命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, dest="logdir")
    args = parser.parse_args()
    return args

def collect_log(logdir):
    """
    遍历指定目录，查找以 'xpu' 开头、'.log' 结尾的文件名
    """
    for filepath, dirnames, filenames in os.walk(logdir):
        for filename in filenames:
            if filename.startswith('xpu') and filename.endswith('.log'):
                yield filename

def loss_check(logname):
    """
    检查训练过程中的损失值，如果损失值与基础值相差较大则返回失败列表
    """
    threshold = 0.001
    fail_msg_list = []
    pass_msg_list = []
    case_id = logname.split('.')[0]
    cmd = f"cat {logname} | grep 'lm loss:' | awk -F 'lm loss: ' '{{print $2}}' | awk '{{print $1}}' > {case_id}"
    execProc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    execProc.wait()
    with open(case_id, 'r') as f:
        loss = f.readlines()
        loss_list = [n.strip() for n in loss]
        step1_loss = float(format(float(loss_list[0]), '.4f'))
        step2_loss = float(format(float(loss_list[1]), '.4f'))
    with open(base_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    for key, value in json_data.items():
        if case_id in key:
            step1_base = float(json_data[key]['step1_loss'])
            step2_base = float(json_data[key]['step2_loss'])
            err1 = abs(step1_base - step1_loss)
            err2 = abs(step2_base - step2_loss)
            print(f"case {case_id} step1_loss is {step1_loss}, step1_base is {step1_base}", \
                  f"step2_loss is {step2_loss}, step2_base is {step2_base}")
            if err1 > threshold or err2 > threshold:
                fail_msg_list.append(1)
            return fail_msg_list
if __name__ == '__main__':
    args = get_args()
    fail_msg = []
    for logname in collect_log(args.logdir):
        msg = loss_check(logname)
        fail_msg.append(msg)
    for sublist in fail_msg:
        assert len(sublist) == 0, f'case failed, please refer to the print info above!'
