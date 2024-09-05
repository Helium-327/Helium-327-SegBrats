
# -*- coding: UTF-8 -*-
'''

Describle:         一些nii.gz格式的处理函数：
                    2. 读取nii.gz文件
                    1. 保存nii.gz文件

Created on         2024/07/31 15:33:54
Author:           @ Mr_Robot
Current State:    
'''
import nibabel as nib
import numpy as np
import os
import h5py
from tqdm import tqdm
import pandas as pd

def red_nii(path):
    pass


def save_nii(data, label):
    """
    保存数据到nii.gz文件
    :param data: 数据
    :param label: 标签
    :return: None
    """
    for i in range(4):
        img = data[i,...].cpu().numpy()
        img = nib.nifti1.Nifti1Image(img, np.eye(4))

        nib.save(img, f"./data_{i}.nii.gz")
    
    label = label.cpu().numpy()
    label = nib.nifti1.Nifti1Image(label.astype(np.float32), np.eye(4))
    nib.save(label, "./label.nii.gz")
    
