# -*- coding: UTF-8 -*-
'''

Describle:         写日志的函数

Created on         2024/08/19 10:46:01
Author:            @ Mr_Robot
Current State:     #TODO:
'''


import os, sys
import time
from tabulate import tabulate
from contextlib import redirect_stdout


def writeTraininglog(save_path, content, file_name):
    '''
    重定向输出到文件
    :param save_path: 保存路径
    :param content: 内容
    :param time_str: 时间
    '''
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Create path: {save_path}")
    with open(os.path.join(save_path, file_name), 'a') as f:
        with redirect_stdout(f):
            print(f"log Time: {now}")
            print(content, end="\n\n")