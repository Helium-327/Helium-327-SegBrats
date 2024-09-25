import subprocess

import socket
import os
import signal


def run_shell_command(command):
    """
    运行shell命令
    """
    print(f"正在运行shell命令: {command}")
    results = subprocess.run(command, shell=True, capture_output=True, text=True)
    if results.returncode == 0:
        print(f"命令执行成功！{results.stdout}")
    else:
        print(f"命令执行失败！, 请查看错误信息：{results.stderr}")
    return results

def start_tensorboard(log_path, PORT=6006, HOST='0.0.0.0', tb=False):
    print(f"正在清理端口占用...")
    command = f"nohup kill -9 $(lsof -t -i:{PORT}) > /dev/null 2>&1 &"
    results = run_shell_command(command)
    print(f"正在启动 tensorboard面板..., log_path: {log_path}")
    if tb:
        command = f"nohup tensorboard --logdir='{log_path}' --port={PORT} --host={HOST} > /dev/null 2>&1 &"
    else:
        command = f"nohup python3 -m tensorboard.main --logdir='{log_path}' --port={PORT} --host={HOST} > /dev/null 2>&1 &"
    results = run_shell_command(command)
    if results.returncode == 0:
        print(f"TensorBoard 启动成功！请通过 localhost:{PORT} 打开")
    else:
        print(f"TensorBoard 启动失败！, 请查看错误信息：{results.stderr}")

if __name__ == "__main__":
    log_path = '/mnt/d/AI_Research/WS-HUB/WS-segBratsWorkflow/Helium-327-SegBrats/results/2024-09-24/【good】19-15-51/tensorBoard/UNet3D_braTS21_2024-09-24 19-15-47'
    start_tensorboard(log_path, PORT=6008, HOST='0.0.0.0', tb=False)