# -*- coding: UTF-8 -*-
'''

Describle:    数据集处理工具，包括：
              1. 划分数据集
              2. 加载数据集

Created on   2024/07/23 15:50:45
@Author:     Mr_Robot
@state:      Done
'''

import os           # 文件操作
import tarfile      # 解压文件
import torch
from torch.utils.data import random_split
from torchvision.transforms import Compose

from readDatasets.BraTS import BraTS21_3d
# device = "cuda" if torch.cuda.is_available() else "cpu"


    
def split_datasets(datasets, train_split=0.8, val_split=0.1, random_seed=42):
    """
    随机划分数据集为训练集、验证集和测试集
    :param datasets:    数据集对象
    :param train_split: 训练集比例，默认0.8
    :param val_split:   验证集比例，默认0.1
    :param random_seed: 随机种子，默认42
    :return:            训练集、验证集和测试集
    """
    test_split = 1 - train_split - val_split
    print(f"训练集：验证集 = {train_split}:{val_split}")
    train_size = int(len(datasets) * train_split)
    val_size = int(len(datasets) * val_split)
    test_size = len(datasets) - train_size - val_size
    
    generator = torch.Generator(device='cpu')
    generator.manual_seed(random_seed)  # 设置生成器的随机种子
    train_set, val_set, test_set = random_split(datasets, # 数据集
                                                [train_size, val_size, test_size], # 划分比例
                                                generator=generator)
                    
    print("数据集已划分完成，其中:\n"
        f"训练集大小：{len(train_set)}\n",
        f"验证集大小：{len(val_set)}\n",
        f"测试集大小：{len(test_set)}")
    return train_set, val_set, test_set
    
def create_load_datasets(dataset, DataLoader, img_trans=None, mask_trans=None, num_workers=0, batch_size=16, shuffle=True):
    """
    加载数据集，返回数据加载器
    :param dataset:     数据集对象
    :param Dataloader:  数据加载器
    :param Trans:       数据增强，默认None
    :param num_workers: 进程数，默认0
    :param batch_size:  批量大小，默认16
    :param shuffle:     是否打乱数据，默认True
    :return:            数据加载器
    """
    new_dataset = []
    
    for vimage, mask in dataset:
        if img_trans:
            vimage = img_trans(vimage)
        if mask_trans:
            mask = mask_trans(mask)
        new_dataset.append((vimage, mask))
    dataset = new_dataset
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=shuffle)
    return data_loader
                             

if __name__ == '__main__':
    num_epochs = 10 
    num_workers = 8
    batch_size = 1
    train_split = 0.8
    val_split = 0.1
    random_seed=42
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    # data_root = "/mnt/g/DATASETS/BraTS21_original_kaggle"
    root = "/root/data/workspace/BraTS_segmentation/data_brats"
    path_data = os.path.join(root, "BraTS2021_Training_Data")

    # transform = transforms.Compose([
    #     RandomCrop3D((144, 128, 128))
    # ])
    datasets = BraTS21_3d(path_data, local_train=True, length=100, data_size = (144,224,224))
    
    # img_trans = Compose([