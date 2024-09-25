# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/09/25 16:50:41
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 运行终端命令
=================================================
'''

import subprocess

import socket
import os
import signal
import time


def run_shell_command(command):
    """
    运行shell命令
    """
    # print(f"正在运行shell命令: {command}")
    results = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, err = results.communicate()
    # 你可以添加以下代码来检测命令是否完成
    while results.poll() is None:
        time.sleep(1)
    if results.returncode == 0:
        print(f"命令执行成功！{output}")
    else:
        print(f"命令执行失败！, 请查看错误信息：{err}")
    return results

def kill_port(PORT):
    print(f"正在清理端口占用...")
    command = f"nohup kill -9 $(lsof -t -i:6007) > /dev/null 2>&1"
    results = run_shell_command(command)
    output, err = results.communicate()
    if results.returncode == 0:
        print(f"🗑️ {PORT} 端口已被清空, {output}")
    else:
        print(f"{err}")
    return results

def start_tensorboard(log_path, PORT=6006, HOST='0.0.0.0', tb=False):
    kill_port(PORT)
    print(f"😃 正在启动 tensorboard面板...\nlog_path: {log_path}")
    if tb:
        command = f"nohup tensorboard --logdir='{log_path}' --port={PORT} --host={HOST} > /dev/null 2>&1 &"
    else:
        command = f"nohup python3 -m tensorboard.main --logdir='{log_path}' --port={PORT} --host={HOST} > /dev/null 2>&1 &"
    results = run_shell_command(command)
    output, err = results.communicate()
    if results.returncode == 0:
        print(f"🆗TensorBoard 启动成功！请通过 localhost:{PORT} 打开, {output}")
    else:
        return f"❌TensorBoard 启动失败！ {err}"

if __name__ == "__main__":
    log_path = '/root/data/workspace/Helium-327-SegBrats/results/2024-09-25/10-46-15/tensorBoard/UNet3D_braTS21_2024-09-25 10-46-11'
    start_tensorboard(log_path, PORT=6007, HOST='0.0.0.0', tb=False)