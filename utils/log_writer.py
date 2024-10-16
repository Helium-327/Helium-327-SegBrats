# -*- coding: UTF-8 -*-
'''

Describle:         写日志的函数

Created on         2024/08/19 10:46:01
Author:            @ Mr_Robot
Current State:     #TODO:
'''
from datetime import datetime
import os

def custom_logger(content, file_pth, log_time=False):
    '''
    自定义日志写入函数
    :param content: 日志内容
    :param file_pth: 日志文件路径
    '''

    now = get_current_date() + " " + get_current_time()
    if not os.path.exists(file_pth):
        with open(file_pth, 'a') as f:
            if log_time:
                f.write(f"log Time: {now}\n")
            f.write(content + '\n')
    else:
        with open(file_pth, 'a') as f:
            if log_time:
                f.write(f"log Time: {now}\n")
            f.write(content + '\n')

# 获取当前日期
def get_current_date():
    return datetime.now().strftime('%Y-%m-%d')

# 获取当前时间
def get_current_time():
    return datetime.now().strftime('%H:%M:%S')