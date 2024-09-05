# -*- coding: UTF-8 -*-
'''

Describle:         损失函数

Created on          2024/07/24 15:24:43
Author:             @ Mr_Robot
Current State:      增加Diceloss，测试通过
'''
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import torch

class DiceLoss:
    def __init__(self, smooth=1e-5):
        """
        初始化函数:
        :param smooth: 平滑因子
        """
        self.smooth = smooth
        self.sub_areas = ['ET', 'TC', 'WT']
        self.labels = {
            'BG': 0, 
            'NCR' : 1,
            'ED': 2,
            'ET':3
        }
        self.num_classes = len(self.labels)

    def __call__(self, y_pred, y_mask):
        """
        DiceLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        loss_dict = {}
        sub_areas = self.sub_areas
        tensor_one = torch.tensor(1)
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # y_mask ==> [batch, 4, 144, 128, 128]
        
        intersection = (y_pred * y_mask).sum(dim=(-3, -2, -1))
        union = y_pred.sum(dim=(-3, -2, -1)) + y_mask.sum(dim=(-3, -2, -1))
        dice = 2 * (intersection + self.smooth) / (union + self.smooth)
        loss = tensor_one - dice.mean()  # 必须是tensor(1)
        
        pred_list, mask_list = splitSubAreas(y_pred, y_mask)
        # 计算子区域的diceloss
        for sub_area, pred, mask in zip(sub_areas, pred_list, mask_list):
            intersection = (pred * mask).sum(dim=(-3, -2, -1))
            union = pred.sum(dim=(-3, -2, -1)) + mask.sum(dim=(-3, -2, -1))
            dice_c = 2 * (intersection + self.smooth) / (union + self.smooth)
            loss_dict[sub_area] = tensor_one - dice_c.mean()
        
        et_loss = loss_dict['ET']
        tc_loss = loss_dict['TC']
        wt_loss = loss_dict['WT']
        

        return loss, et_loss, tc_loss, wt_loss


class CELoss:
    def __init__(self, smooth=1e-5):
        """
        初始化函数:
        :param smooth: 平滑因子
        """
        self.smooth = smooth
        self.sub_areas = ['ET', 'TC', 'WT']
        self.labels = {
            'BG': 0, 
            'NCR' : 1,
            'ED': 2,
            'ET':3
        }
        self.num_classes = len(self.labels)
    def __call__(self, y_pred, y_mask):
        """
        CrossEntropyLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        loss_dict = {}
        sub_areas = self.sub_areas
        
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        
        pred_list, mask_list = splitSubAreas(y_pred, y_mask)
        
        for sub_area, pred, mask in zip(sub_areas, pred_list, mask_list):
            loss = CrossEntropyLoss()(pred, mask)
            loss_dict[sub_area] = loss
        
        et_loss = loss_dict['ET']
        tc_loss = loss_dict['TC']
        wt_loss = loss_dict['WT']
        mean_loss = sum(loss_dict.values()) / len(sub_areas)
        
        return mean_loss, et_loss, tc_loss, wt_loss        


def splitSubAreas(y_pred, y_mask):
    """
    分割出子区域
    :param y_pred: 预测值 [batch, 4, D, W, H]
    :param y_mask: 真实值 [batch, 4, D, W, H]
    """
    et_pred = y_pred[:, 3,...]
    tc_pred = y_pred[:, 1,...] + y_pred[:,3,...]
    wt_pred = y_pred[:, 1:,...].sum(dim=1)
    
    et_mask = y_mask[:, 3,...]
    tc_mask = y_mask[:, 1,...] + y_mask[:,3,...]
    wt_mask = y_mask[:, 1:,...].sum(dim=1)
    
    pred_list = [et_pred, tc_pred, wt_pred]
    mask_list = [et_mask, tc_mask, wt_mask]
    return pred_list, mask_list
        

if __name__ == '__main__':
    y_pred = torch.randn(1, 4, 144, 128, 128)
    # y_pred = torch.argmax(y_pred, dim=1)
    y_mask = torch.randint(0, 4, (1, 144, 128, 128))
    diceLossFunc = DiceLoss()
    ceLossFunc = CELoss()

    print(f"CELoss : {diceLossFunc(y_pred, y_mask)}")
    
    print(f"DiceLoss : {ceLossFunc(y_pred, y_mask)}")
    