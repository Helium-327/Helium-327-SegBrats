# -*- coding: UTF-8 -*-
'''

代码说明:    训练流程

Created on      2024/07/23 15:28:23
Author:         @Mr_Robot
State:          3d can run 
'''

import os
import time
import torch
import numpy as np
# from torch.nn import CrossEntropyLoss
# from loss_function import Diceloss, crossEntropy_loss
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

# from torch.optim import Adam, SGD, RMSprop, AdamW
from torch.amp import GradScaler, autocast

# 加载本地模块
from readDatasets.BraTS import BraTS21_3d
from nets.unet3d import UNet_3D
# from loss_function import DiceLoss, CELoss
from utils.datasets import split_datasets    # 这个导入怎么导的？
from metrics import EvaluationMetrics

os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
best_val_loss = float('inf')
start_epoch = 0

def train_one_epoch(model, train_loader, scaler, optimizer, loss_function, device):
    """
    ====训练过程====
    :param model: 模型
    :param metrics: 评估指标
    :param train_loader: 训练数据集
    :param val_loader: 验证数据集
    :param scaler: 缩放器
    :param optimizer: 优化器
    :param loss_funtion: 损失函数
    :param device: 设备
    :param model_path: 模型路径
    """
    model.train()
    
    train_running_loss = 0.0
    
    train_et_loss = 0.0
    train_tc_loss = 0.0
    train_wt_loss = 0.0
    mean_train_et_loss = 0.0
    mean_train_tc_loss = 0.0
    mean_train_wt_loss = 0.0
    
    train_loader = tqdm(train_loader, desc=f"# Training", leave=False)
    
    for data in train_loader: # 读取每一个 batch
        # 获取输入数据
        vimage, mask = data[0].to(device), data[1].to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        with autocast(device_type='cuda'): # 混合精度训练
            # 前向传播 + 反向传播 + 优化
            predicted_mask = model(vimage)
            mean_loss, et_loss, tc_loss, wt_loss = loss_function(predicted_mask, mask)
        scaler.scale(mean_loss).backward()                           # 反向传播，只有训练模型时才需要
        scaler.step(optimizer)                                  # 优化器更新参数
        scaler.update()  
        
        train_running_loss += mean_loss.item()                       # 计算训练loss的累计和
        train_et_loss += et_loss.item() 
        train_tc_loss += tc_loss.item()
        train_wt_loss += wt_loss.item()
        
    return train_running_loss, train_et_loss, train_tc_loss, train_wt_loss

def val_one_epoch(model, Metric, val_loader, loss_function, epoch, device):
    """
    验证过程
    :param model: 模型
    :param metrics: 评估指标
    :param train_loader: 训练数据集
    :param val_loader: 验证数据集
    :param scaler: 缩放器
    :param optimizer: 优化器
    :param loss_funtion: 损失函数
    :param device: 设备
    :param model_path: 模型路径
    """
    val_running_loss = 0.0
    Metrics_list = np.zeros((7, 4))
    model.eval()
    val_et_loss = 0.0
    val_tc_loss = 0.0
    val_wt_loss = 0.0
    # mean_val_et_loss = 0.0
    # mean_val_tc_loss = 0.0
    # mean_val_wt_loss = 0.0
    
    with torch.no_grad(): # 关闭梯度计算
        with autocast(device_type='cuda'):
            val_loader = tqdm(val_loader, desc=f"== Validating ==", leave=False)
            for data in val_loader:
                vimage, mask = data[0].to(device), data[1].to(device)                
                with autocast(device_type='cuda'):
                    predicted_mask = model(vimage)
                    mean_loss, et_loss, tc_loss, wt_loss = loss_function(predicted_mask, mask)
                    metrics = Metric.update(predicted_mask, mask)
                    Metrics_list += metrics
                val_running_loss += mean_loss.item() 
                val_et_loss += et_loss.item() 
                val_tc_loss += tc_loss.item()
                val_wt_loss += wt_loss.item()
                
    Metrics_list /= len(val_loader)
    
        
    return val_running_loss, val_et_loss, val_tc_loss, val_wt_loss, Metrics_list

    

if __name__ == "__main__":
    
    # num_epochs = 10 
    # num_workers = 8
    # batch_size = 1
    # train_split = 0.8
    # val_split = 0.1
    # random_seed=42
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    # # data_root = "/mnt/g/DATASETS/BraTS21_original_kaggle"
    # root = "/root/data/workspace/BraTS_segmentation/data_brats"
    # path_data = os.path.join(root, "BraTS2021_Training_Data")

    # # transform = transforms.Compose([
    # #     RandomCrop3D((144, 128, 128))
    # # ])
    # datasets = BraTS21_3d(path_data, local_train=True, length=10, data_size = (144,224,224))
    
    # # datasets_local = BraTS21_2d(local_dir,"t1", local_train=True, lenth=50)
    
    # model = UNet_3D(4, 4)
    # model.to(device)
    
    # val_metrics = EvaluationMetrics()
    
    # train_datasets, val_datasets, test_datasets = split_datasets(datasets, 
    #                                                         train_split=train_split,
    #                                                         val_split=val_split, 
    #                                                         random_seed=random_seed)
    # train_loader = DataLoader(train_datasets, 
    #                         batch_size = batch_size,
    #                         num_workers = num_workers,
    #                         shuffle=True)
    
    # val_loader = DataLoader(val_datasets, 
    #                         batch_size = batch_size,
    #                         num_workers=num_workers,
    #                         shuffle=False)
    # test_loader = DataLoader(test_datasets, 
    #                         batch_size = batch_size,
    #                         num_workers=num_workers,
    #                         shuffle=False)
    
    # # optimizer = Adam(model.parameters(), lr=1e-3)

    # # 使用SGD优化器
    # optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # scaler = GradScaler()
    
    # # 使用RMSPROP优化器
    # # optimizer = RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1e-8)
 
    # # 使用AdamW优化器
    # # optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # # loss_function = Diceloss()

    # loss_function = LossFunctions()

    # train_and_val(model, val_metrics, train_loader, val_loader, scaler, optimizer,  loss_function, num_epochs, device)

    pass