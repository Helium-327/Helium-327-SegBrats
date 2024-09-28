# -*- coding: UTF-8 -*-
'''

Describle:         读取数据列表，分割为训练集和测试集

Created on         2024/08/03 14:13:05
Author:            @ Mr_Robot
Current State:     #TODO:
'''
import os
import random
import pandas as pd

class dataSpliter:
    def __init__(self, data_root, train_split=0.8, val_split=0.1, seed=42):
        self.data_root = data_root
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1 - self.train_split - self.val_split
        self.seed = seed

    def get_data_list(self):
        data_list = []
        data_root = self.data_root

        for root, dirs, _ in os.walk(data_root):
            for patient_idx in dirs:            # 遍历病例文件夹
                patient_dict = {}
                patient_dir = os.path.join(root, patient_idx)  # 获取病例文件夹路径
                patient_dict['patient_idx'] = patient_idx
                patient_dict['patient_dir'] = patient_dir
                data_list.append(patient_dict)
        return data_list

    def random_data_list(self):
        data_list = self.get_data_list()
        if self.seed:
            random.seed(self.seed)
        random.shuffle(data_list)

        return data_list

    def data_split(self):
        # data_list = self.get_data_list()
        data_list_shuffled = self.random_data_list()
        data_length = len(data_list_shuffled)
        train_length = int(data_length * self.train_split)
        val_split_length = int(data_length * self.val_split)

        train_list = data_list_shuffled[:train_length]
        test_list = data_list_shuffled[train_length:train_length+val_split_length]
        val_list = data_list_shuffled[train_length+val_split_length:]

        return train_list, test_list, val_list
    
    def save_as_csv(self, data_list, csv_path):
        df = pd.DataFrame(data_list)
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")

if __name__ == '__main__':
   
    root = "/mnt/d/AI_Research/WS-HUB/WS-segBratsWorkflow/SegBrats3d/brats21_local" # TODO: 修改路径(使用绝对路径)
    path_data = os.path.join(root, "BraTS2021_Training_Data")
    
    assert os.path.exists(root), f"{root} not exists."
    
    dataspliter =  DataSpliter(path_data, train_split=0.8, val_split=0.1)

    train_list, test_list, val_list = dataspliter.data_split()
    dataspliter.save_as_csv(train_list, os.path.join(root, "train.csv"))
    dataspliter.save_as_csv(test_list, os.path.join(root, "test.csv"))
    dataspliter.save_as_csv(val_list, os.path.join(root, "val.csv"))