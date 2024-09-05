# -*- coding: UTF-8 -*-
'''

Describle:         BraTS数据集类, 用于加载数据集, 切分数据集

Created on         2024/07/23 16:18:28
Author:            @ Mr_Robot
Current State:     To be Refine
'''

import os
import pandas as pd
# from re import L
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.functional import one_hot

# from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt



    
    
    
# ========================= 3D 数据集加载 ======================================================
        
class BraTS21_3d(Dataset):
    """
    通过预先分好的数据文件路径列表，加载数据集中的病例，尝试加载所有模态图像
    """
    def __init__(self, data_file, transform=None, local_train=False, length=None):
        """
        BraTS数据集初始化
        :param data_dile: 数据集路径列表文件 csv或txt
        :param loacl_train: 是否仅加载部分数据集（可选，默认为False）
        :param length: 加载数据集的长度（可选，默认为None）
        """
        self.data_file = data_file
        self.local_train = local_train
        self.length = length
        self.transform = transform
        self.patients_dirs, self.patients_ids = self.load_patient()
        
    def __len__(self):
        """
        返回数据集中的病例数量
        """
        return len(self.patients_ids)
    
    def load_patient(self, data_file=None):
        """
        通过数据路径列表加载数据集
        :param data_dir: 数据集路径
        :return: 病例文件夹地址列表 # 返回路径列表方便后续重复使用
        """
        if data_file:
            df = pd.read_csv(data_file)
        else:
            df = pd.read_csv(self.data_file)

        if self.local_train:
            df = df.iloc[:self.length] # 部分训练时，加载部分数据集

        patients_dirs = []
        patients_ids = []
        for i in range(len(df)):
            patient_id = df.iloc[i]['patient_idx']
            patient_dir = df.iloc[i]['patient_dir']
            patients_dirs.append(patient_dir)
            patients_ids.append(patient_id)
        return patients_dirs, patients_ids
    
    def load_image(self, patient_dir): # FIXME: 需要将 img 和 label 分开
        """
        加载指定病例和指定模态的图像
        :param patient_dir: 病例文件夹路径
        :param self.modality: 图像模态名称 ("t1", "t1ce", "flair", "t2", "seg")
        :return: 加载的图像(numpy数组)
        """
        # modality = self.modality
        patient_idx = patient_dir.split("/")[-1]
        modalities = ("t1", "t1ce", "flair", "t2", "seg")
        
        t1_path = os.path.join(patient_dir, f"{patient_idx}_t1.nii.gz")
        t1ce_path = os.path.join(patient_dir, f"{patient_idx}_t1ce.nii.gz")
        t2_path = os.path.join(patient_dir, f"{patient_idx}_t2.nii.gz")
        flair_path = os.path.join(patient_dir, f"{patient_idx}_flair.nii.gz")
        seg_path = os.path.join(patient_dir, f"{patient_idx}_seg.nii.gz")
        
        assert os.path.exists(t1_path), f"{t1_path}不存在"
        assert os.path.exists(t1ce_path), f"{t1ce_path}不存在"
        assert os.path.exists(t2_path), f"{t2_path}不存在"
        assert os.path.exists(flair_path), f"{flair_path}不存在"
        assert os.path.exists(seg_path), f"{seg_path}不存在"
        
        t1_data = np.transpose(nib.load(t1_path).get_fdata(), (2, 0, 1))
        t1ce_data = np.transpose(nib.load(t1ce_path).get_fdata(), (2, 0, 1))
        t2_data = np.transpose(nib.load(t2_path).get_fdata(), (2, 0, 1))
        flair_data = np.transpose(nib.load(flair_path).get_fdata(), (2 ,0 ,1))
        mask = np.transpose(nib.load(seg_path).get_fdata(), (2, 0, 1)) 

            
        multiModal_list = [t1_data, t1ce_data, t2_data, flair_data]
        vimage = np.stack(multiModal_list, axis=0)
        vimage = torch.tensor(vimage, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)
        mask[mask == 4] = 3  # 与数据集特性有关
        
        if self.transform:
            vimage, mask = self.transform(vimage, mask)
        return vimage, mask
    
    def __getitem__(self, index):     
        """
        通过索引获取数据集中的病例，尝试加载所有模态图像
        # :param index: 病例索引，从0开始
        :param patient_idx: 病例名称缩写（如：0 ——-> 00000 可选，默认为None）
        :return: 病例数据字典, 包含所有模态图像, key: 模态名称, value: 图像(numpy数组)
        """
        
        patient_dir = self.patients_dirs[index]
        patient_vimage, patient_mask = self.load_image(patient_dir)
                    
        return patient_vimage, patient_mask   
 


if __name__ == "__main__":
    data_root = "/mnt/g/DATASETS/BraTS21_original_kaggle"
    data_dir = os.path.join(data_root, "BraTS2021_00621")

    data_size = (144, 224, 224)
    # transform = RandomCrop3D(target_size)
    # brats = BraTS21_3d(data_dir, data_size=data_size)

    train_dataset = BraTS21_3d("./train.csv")

    print(train_dataset[0][0].shape, train_dataset[0][1].shape)
    # data, label = brats.load_image(data_dir)

    # print(data.shape, label.shape)
    # for i in range(4):
    #     img = data[i,...].cpu().numpy()
    #     img = nib.nifti1.Nifti1Image(img, np.eye(4))

    #     nib.save(img, f"./data_{i}.nii.gz")
    
    # label = label.cpu().numpy()
    # label = nib.nifti1.Nifti1Image(label.astype(np.float32), np.eye(4))
    # nib.save(label, "./label.nii.gz")