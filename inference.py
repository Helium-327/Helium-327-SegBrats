# -*- coding: UTF-8 -*-
'''

Describle:         模型评估：在训练完成之后使用测试集对模型性能进行评估，并保存评估结果

Created on         2024/08/18 14:07:26
Author:            @ Mr_Robot
Current State:     #TODO:
'''
import os
import torch
from tabulate import tabulate
import numpy as np
import nibabel as nib

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torch.amp import GradScaler, autocast

from utils.writinglog import writeTraininglog
from nets.unet3d import UNet_3D
from metrics import EvaluationMetrics
from loss_function import DiceLoss, CELoss
from transforms import data_transform, Compose, RandomCrop3D, Normalize, tioRandomFlip3d
from readDatasets.BraTS import BraTS21_3d
from utils.utils_ckpt import save_checkpoint, load_checkpoint
from utils.splitDataList import DataSpliter
from matplotlib import pyplot as plt

from torch.nn import functional as F

def inference(test_loader, model, Metricer, output_path, device, affine, window_size=(128, 128, 128), stride_size=(13, 32, 32)):
    """
    验证过程：
        1. 使用滑窗预测算法对测试集数据进行推理预测
        2. 保存预测结果nii文件
        3. 使用评估指标对预测结果进行评估
        4. 保存评估结果
    :param model: 模型
    :param metrics: 评估指标
    :param train_loader: 训练数据集
    :param test_loader: 验证数据集
    :param scaler: 缩放器
    :param optimizer: 优化器
    :param loss_funtion: 损失函数
    :param device: 设备
    :param model_path: 模型路径
    """
    Metrics_list = np.zeros((7, 4))
    
    for i, data in enumerate(test_loader):
        vimage, vmask = data[0], data[1]
        # print(vmask.shape, vmask.shape)
        predvimage = slide_window_pred(model, vimage, device, window_size=window_size, stride_size=stride_size)
        print(predvimage.shape)

        # 降维，选出概率最大的类索引值
        test_output_argmax = torch.argmax(predvimage, dim=1).to(dtype=torch.int64) 

        # # 获取one-hot编码,并转置为(batch, C, D, H, W)
        # test_output = F.one_hot(test_output_argmax, num_classes=4).permute(0, 4, 1, 2, 3).float() # one-hot
        # test_mask = F.one_hot(vmask, num_classes=4).permute(0, 4, 1, 2, 3).float() # ont-hot

        num = 0

        # 获取测试集的输入图像和预测输出的图像数据
        save_input_t1 = vimage[num, 0, ...].permute(2, 1, 0).cpu().detach().numpy().astype(np.float32)
        save_input_t1ce = vimage[num, 1, ...].permute(2, 1, 0).cpu().detach().numpy().astype(np.float32)
        save_input_t2 = vimage[num, 2, ...].permute(2, 1, 0).cpu().detach().numpy().astype(np.float32)
        save_input_flair = vimage[num, 3, ...].permute(2, 1, 0).cpu().detach().numpy().astype(np.float32)
        save_input_mask = vmask[num, ...].permute(2, 1, 0).cpu().detach().numpy().astype(np.int16)
        save_pred = test_output_argmax[num,...].permute(2, 1, 0).cpu().detach().numpy().astype(np.int16)


        # 将数据转成nib的对象
        nii_input_t1 = nib.nifti1.Nifti1Image(save_input_t1, affine=affine)
        nii_input_t1ce = nib.nifti1.Nifti1Image(save_input_t1ce, affine=affine)
        nii_input_t2 = nib.nifti1.Nifti1Image(save_input_t2, affine=affine)
        nii_input_flair = nib.nifti1.Nifti1Image(save_input_flair, affine=affine)
        nii_input_mask = nib.nifti1.Nifti1Image(save_input_mask, affine=affine)
        nii_pred = nib.nifti1.Nifti1Image(save_pred, affine=affine)

        # 保存nii文件
        nib.save(nii_input_t1, os.path.join(output_path, f'P{i}_test_input_t1.nii.gz'))
        nib.save(nii_input_t1ce, os.path.join(output_path, f'P{i}_test_input_t1ce.nii.gz'))
        nib.save(nii_input_t2, os.path.join(output_path, f'P{i}_test_input_t2.nii.gz'))
        nib.save(nii_input_flair, os.path.join(output_path, f'P{i}_test_input_flair.nii.gz'))
        nib.save(nii_input_mask, os.path.join(output_path, f'P{i}_test_input_mask.nii.gz'))
        nib.save(nii_pred, os.path.join(output_path, f'P{i}_test_pred.nii.gz'))

        print(f"P{i} pred save successfully! path on {output_path}")


        # 评估指标
        
        metrics = Metricer.update(predvimage, vmask)
        Metrics_list += metrics

    Metrics_list /= len(test_loader)

    test_scorce = {}
    # 记录验证结果
    test_scorce['Dice_scores'] = Metrics_list[0] 
    test_scorce['Jaccard_scores'] = Metrics_list[1]
    test_scorce['Accuracy_scores'] = Metrics_list[2]
    test_scorce['Precision_scores'] = Metrics_list[3]
    test_scorce['Recall_scores'] = Metrics_list[4]
    test_scorce['F1_scores'] = Metrics_list[5]
    test_scorce['F2_scores'] = Metrics_list[6]
    metric_table_header = ["Metric_Name", "MEAN", "ET", "TC", "WT"]
    metric_table_left = ["Dice", "Jaccard", "Accuracy", "Precision", "Recall", "F1", "F2"]

    # 优化点：直接通过映射获取指标名称，避免重复字符串格式化
    def format_value(value, decimals=4):
        # 返回一个格式化后的字符串，保留指定的小数位数
        return f"{value:.{decimals}f}"
    
    metric_scores_mapping = {metric: test_scorce[f"{metric}_scores"] for metric in metric_table_left}
    metric_table = [[metric,
                    format_value(metric_scores_mapping[metric][0]),
                    format_value(metric_scores_mapping[metric][1]),
                    format_value(metric_scores_mapping[metric][2]),
                    format_value(metric_scores_mapping[metric][3])] for metric in metric_table_left]
    table_str = tabulate(metric_table, headers=metric_table_header, tablefmt='grid')
    metrics_info = table_str

    log_name = f"test_metrics.txt"
    writeTraininglog(output_path, metrics_info, log_name)
    print(metrics_info)

def slide_window_pred(model, test_data, device, window_size, stride_size):
    N, C, D, H, W = test_data.shape
    model.eval()

    with torch.no_grad():
        with autocast(device_type='cuda'):
            pred_mask = torch.zeros_like(test_data)
            for d in range(0, D - window_size[0]+1, stride_size[0]): # D 维度
                for h in range(0, H - window_size[1]+1, stride_size[1]): # H 维度
                    for w in range(0, W - window_size[2]+1, stride_size[2]): # W 维度
                        # print(predvimage != test_data)
                        patch = test_data[:, :, d:d+window_size[0], h:h+window_size[1], w:w+window_size[2]]
                        patch = patch.to(device)

                        pred = model(patch)
                        # pred = pred.cpu().numpy()

                        # 将预测结果保存到原始图像的对应位置
                        pred_mask[:, :, d:d+window_size[0], h:h+window_size[1], w:w+window_size[2]] = pred

    return pred_mask

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = r"/mnt/d/AI_Research/WS-HUB/WS-BraTS/BraTS_segmentation/results/outputs"
    checkpoint_path = r"/mnt/g/BraTS实验结果/2024-8-27/checkpoints/BraTS21_3d_2024-08-25-16-07-51_364_0.8973.pth"
    test_csv = "/mnt/d/AI_Research/WS-HUB/WS-BraTS/BraTS_segmentation/brats21_local/val.csv"

    affine = np.array([[ -1.,  -0.,  -0.,   0.],
                    [ -0.,  -1.,  -0., 239.],
                    [  0.,   0.,   1.,   0.],
                    [  0.,   0.,   0.,   1.]])
    

    if not os.path.exists(output_path):
        print("output path not exists, create it")
        os.makedirs(output_path)
    

    model = UNet_3D(in_channels=4, num_classes=4)
    model.to(device)
    
    Metricer = EvaluationMetrics()
    # loss_function = LossFunctions()

    TransMethods_val   = data_transform(transform=Compose([RandomCrop3D(size=(154, 224, 224)),    # 随机裁剪
                                                        # tioRandonCrop3d(size=CropSize),
                                                         Normalize(mean=(0.114, 0.090, 0.170, 0.096), std=(0.199, 0.151, 0.282, 0.174)),   # 标准化
                                                         tioRandomFlip3d(),   
                                                        # tioRandomAffine(),          # 随机旋转
                                                        
                                                        # tioRandomFlip3d(),                 # 随机翻转
                                                        # tioRandomElasticDeformation3d(),
                                                        # tioZNormalization(),               # 归一化
                                                        # tioRandomNoise3d(),
                                                        # tioRandomGamma3d()    
                                        ]))

    test_dataset  = BraTS21_3d(test_csv, 
                            transform=TransMethods_val, 
                            local_train=True, 
                            length=5)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=1, 
                            num_workers=0, 
                            shuffle=False)
    

    optimizer = RMSprop(model.parameters(), lr=1e-3, alpha=0.9, eps=1e-8)
    scaler = GradScaler()


    model, optimizer, scaler, start_epoch, best_val_loss = load_checkpoint(model, optimizer, scaler, checkpoint_path)
    # print(model)
    inference(test_loader, model, Metricer, output_path, device, affine)