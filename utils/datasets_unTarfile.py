# -*- coding: UTF-8 -*-
'''

Describle:         一些文件操作的工具：
                    1. 解压tar文件

Created on         2024/07/31 15:34:22
Author:           @ Mr_Robot
Current State:    
'''
import os           # 文件操作
import tarfile      # 解压文件
import torch
from torch.utils.data import random_split


def unTarfile(tar_path, save_path):
    '''
    :param tar_path: 待解压文件路径
    :param save_path: 解压后保存的路径
    :return:
    '''
    if os.path.exists(save_path):
        print("目标文件夹已存在，无需解压")
    else:
        os.makedirs(save_path)
        print("目标文件夹不存在，已创建文件夹，正在解压...")
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=save_path)
                tar.close()
                print(f"已解压成功，文件路径为: {save_path}")
        except tarfile.TarError as e:
            print(f"解压文件时发生错误: {e}")
        except FileNotFoundError as e:
            print(f"文件未找到: {e}")
        except PermissionError as e:
            print(f"权限错误: {e}")
        except Exception as e:
            print(f"未知错误: {e}")