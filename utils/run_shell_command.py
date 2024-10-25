# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/09/25 16:50:41
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 运行终端命令
=================================================
'''

import subprocess
import time

def run_shell_command(command):
    """
    运行shell命令
    """
    try:
        # print(f"正在运行shell命令: {command}")
        results = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, err = results.communicate()
        if results.returncode == 0:
            print(f"命令执行成功！{output}")
        else:
            print(f"命令执行失败！, 请查看错误信息：{err}")
        return results.returncode == 0
    except Exception as e:
        print(f"执行命令时发生异常：{e}")
        return False

def is_port_is_used(PORT):
    """
    检查端口是否被占用
    """
    command = f"lsof -i :{PORT}"
    return run_shell_command(command)

def kill_port(PORT):
    """
    清理端口占用
    """
    if not is_port_is_used(PORT):
        print(f"端口{PORT}未被占用，无需清理。")
        return True
    else:
        command = f"lsof -t -i:{PORT} | xargs kill -9"
        print(f"正在清理端口占用...")
        return run_shell_command(command)

def start_tensorboard(log_path, PORT=6006, HOST='0.0.0.0'):
    """
    启动TensorBoard面板
    确保是在cv环境下运行，否则无法启动
    """
    if not kill_port(PORT):
        return(f"清理端口{PORT}失败，启动TensorBoard失败。")      
    else:
        print(f"😃 正在启动 tensorboard面板...\nlog_path: {log_path}")
        command = f"nohup python3 -m tensorboard.main --logdir='{log_path}' --port={PORT} --host={HOST} > /dev/null 2>&1 &"
        if not run_shell_command(command):
            print(f"TensorBoard 启动失败！")
        else:
            time.sleep(5)
            print(f"😃 TensorBoard 启动成功！\n请访问 localhost:{PORT} 查看TensorBoard面板。")

if __name__ == "__main__":
    log_path = '/root/workspace/Helium-327-SegBrats/results/2024-10-25/2024-10-25_15-31-20/tensorBoard'
    start_tensorboard(log_path, PORT=6006, HOST='0.0.0.0')