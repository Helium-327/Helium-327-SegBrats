# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/09/21 10:32:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 实验开始之前记录当前实验的内容在开始实验
=================================================
'''

import os
from datetime import datetime



# 获取当前日期
def get_current_date():
    return datetime.now().strftime('%Y-%m-%d')

# 获取当前时间
def get_current_time():
    return datetime.now().strftime('%H-%M-%S')

def write_commit_file(file_path, content):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("# UNet3d brats实验记录\n\n")
            f.write("## 实验记录\n")
    with open(file_path, 'a') as f:
        f.write(f"- **Commit on:**`{get_current_date() +'-' + get_current_time()}`\n")
        f.write(f"  > {content}\n")
        
# 根据当前日期创建文件夹
def create_folder(folder_path):
    if os.path.exists(folder_path):
        folder_path = os.path.join(folder_path, get_current_time())
        os.makedirs(folder_path)
        print(f"当前日期文件夹已存在，将创建时间文件夹：{folder_path} ")
    else: # 如果日期文件夹不存在，则先创建日期文件夹， 再创建时间文件夹
        os.makedirs(folder_path)
        print(f"当前日期文件夹 '{folder_path}' 已经创建。")
        
        folder_path = os.path.join(folder_path, get_current_time())
        os.makedirs(folder_path)
        print(f"当前时间文件夹 '{folder_path}' 已经创建。")

    return folder_path
if __name__ == "__main__":
    # 提示用户输入本次实验内容
    content = input("请输入本次实验内容：")

    # 创建结果保存路径
    results_dir = os.path.join('./results', get_current_date())
    results_dir = create_folder(results_dir)

    # 写入实验记录
    write_commit_file(os.path.join(results_dir,'实验记录.md'), content)