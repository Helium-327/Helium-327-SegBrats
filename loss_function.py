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
    def __init__(self, loss_type='subarea_custom', smooth=1e-5, w_bg=0.2, w1=0.2, w2=0.2, w3=0.4):
        """
        初始化函数:
        :param smooth: 平滑因子
        :param w1: ET权重
        :param w2: TC权重
        :param w3: WT权重
        """
        self.smooth = smooth
        self.loss_type = loss_type
        self.sub_areas = ['BG', 'ET', 'TC', 'WT']
        self.labels = {
            'BG': 0, 
            'NCR' : 1,
            'ED': 2,
            'ET':3
        }
        self.num_classes = len(self.labels)
        self.w1, self.w2, self.w3, self.w_bg = w1, w2, w3, w_bg

    def __call__(self, y_pred, y_mask):
        """
        DiceLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        tensor_one = torch.tensor(1)
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float() # y_mask ==> [batch, 4, 144, 128, 128]
        area_bg_loss, area_et_loss, area_tc_loss, area_wt_loss = self.get_every_subAreas_loss(y_pred, y_mask)

        assert self.loss_type in ['subarea_custom', 'classes_custom', 'mean'], f'loss_type must be in ["custom", "mean"], but got {self.loss_type}'
        if self.loss_type == 'subarea_custom':          # FIXME: 似乎存在重复计算，有待优化
            bg_loss, et_loss, tc_loss, wt_loss = area_bg_loss, area_et_loss, area_tc_loss, area_wt_loss
            loss = self.w_bg * bg_loss + self.w1 * wt_loss + self.w2 * tc_loss + self.w3 * et_loss

        elif self.loss_type =='classes_custom':         #TODO: 为每一类loss单独设置权重，消除类不平衡对于损失函数的影响
            bg_loss, ncr_loss, ed_loss, et_loss = self.get_every_class_loss(y_pred, y_mask)
            loss = self.w_bg * bg_loss + self.w1 * ed_loss + self.w2 * ncr_loss + self.w3 * et_loss
        else:
            intersection = (y_pred * y_mask).sum(dim=(-3, -2, -1))
            union = y_pred.sum(dim=(-3, -2, -1)) + y_mask.sum(dim=(-3, -2, -1))
            dice = 2 * (intersection ) / (union + self.smooth)
            mean_loss = tensor_one - dice.mean(dim=1)   # 必须是tensor(1)
            loss = mean_loss.mean()             

        return loss, area_et_loss, area_tc_loss, area_wt_loss

    def get_every_subAreas_loss(self, y_pred, y_mask):
        loss_dict = {}
        pred_list, mask_list = splitSubAreas(y_pred, y_mask)

        # 计算子区域的diceloss
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            intersection = (pred * mask).sum(dim=(-3, -2, -1))
            union = pred.sum(dim=(-3, -2, -1)) + mask.sum(dim=(-3, -2, -1))
            dice_c = 2 * (intersection) / (union + self.smooth)
            loss_dict[sub_area] = 1 - dice_c.mean()

        # 计算batch平均损失
        area_bg_loss = loss_dict['BG']
        area_et_loss = loss_dict['ET']
        area_tc_loss = loss_dict['TC']
        area_wt_loss = loss_dict['WT']

        return area_bg_loss, area_et_loss, area_tc_loss, area_wt_loss
    
    def get_every_class_loss(self, y_pred, y_mask):
        loss_dict = {}

        # 获取子类别的预测值和真实值
        pred_list, mask_list = splitClasses( y_pred, y_mask)

        #  计算子类别损失
        classes_list = [k for k in self.labels.keys()]
        for subclass, pred, mask in zip(classes_list, pred_list, mask_list):
            intersection = (pred * mask).sum(dim=(-3, -2, -1))
            union = pred.sum(dim=(-3, -2, -1)) + mask.sum(dim=(-3, -2, -1))
            dice_c = 2 * (intersection) / (union + self.smooth)
            loss_dict[subclass] = 1 - dice_c.mean()

        # 计算batch平均损失
        class_bg_loss = loss_dict['BG']
        class_ncr_loss = loss_dict['NCR']
        class_ed_loss = loss_dict['ED']
        class_et_loss = loss_dict['ET']

        return class_bg_loss, class_ncr_loss, class_ed_loss, class_et_loss


# Focal Loss
class FocalLoss:
    def __init__(self, loss_type='mean', gamma=2, alpha=0.8, w1=0.3, w2=0.3, w3=0.3, w_bg=0.1):
        """
        初始化函数:
        :param loss_type: loss计算方式，可选['custom', 'mean']，默认为'custom'
        :param gamma: 
        :param alpha: 
        :param w1: ET权重
        :param w2: TC权重
        :param w3: WT权重
        """
        self.loss_type = loss_type
        self.gamma = gamma
        self.alpha = alpha
        self.sub_areas = ['BG', 'ET', 'TC', 'WT']
        self.labels = {
            'BG': 0, 
            'NCR' : 1,
            'ED': 2,
            'ET':3
        }
        self.num_classes = len(self.labels)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w_bg = w_bg
        
    def __call__(self, y_pred, y_mask):
        """
        FocalLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        
        """
        loss_dict = {}
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        mean_loss = self.cal_focal_loss(y_pred, y_mask) 

        pred_list, mask_list = splitSubAreas(y_pred, y_mask)
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            loss = self.cal_focal_loss(pred, mask)
            loss_dict[sub_area] = loss 
        
        bg_loss = loss_dict['BG']
        et_loss = loss_dict['ET']
        tc_loss = loss_dict['TC']
        wt_loss = loss_dict['WT']
        custom_loss = self.w1 * et_loss + self.w2 * tc_loss + self.w3 * wt_loss + self.w_bg * bg_loss
        assert self.loss_type in ['custom', 'mean'], f'loss_type must be in ["custom", "mean"], but got {self.loss_type}'
        if self.loss_type == 'custom':
            global_loss = custom_loss
        else:
            global_loss = mean_loss

        return global_loss, et_loss, tc_loss, wt_loss
    def cal_focal_loss(self, y_pred, y_mask):
        cross_entropy = F.cross_entropy(y_pred, y_mask, reduction="mean")
        pt = torch.exp(-cross_entropy)
        focal_loss = (self.alpha * ((1 - pt) ** self.gamma) * cross_entropy)
        return focal_loss
    
# CELoss
class CELoss:
    def __init__(self, loss_type='mean', smooth=1e-5, w1=0.3, w2=0.3, w3=0.3, w_bg=0.1):
        """
        初始化函数:
        :param smooth: 平滑因子
        :param w1: ET权重
        :param w2: TC权重
        :param w3: WT权重
        """
        self.smooth = smooth
        self.loss_type = loss_type
        self.sub_areas = ['BG', 'ET', 'TC', 'WT'] # 异常子区域只有后三个
        self.labels = {
            'BG': 0,  # 背景
            'NCR' : 1, # 
            'ED': 2, # 增强肿瘤
            'ET':3
        }
        self.num_classes = len(self.labels)

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w_bg = w_bg

    def __call__(self, y_pred, y_mask):
        """
        CrossEntropyLoss
        :param y_pred: 预测值 [batch, 4, D, W, H]
        :param y_mask: 真实值 [batch, D, W, H]
        """
        loss_dict = {}
        
        y_mask = F.one_hot(y_mask, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        pred_list, mask_list = splitSubAreas(y_pred, y_mask)
        
        global_CEloss = F.cross_entropy(y_pred, y_mask, reduction="mean")
        for sub_area, pred, mask in zip(self.sub_areas, pred_list, mask_list):
            CEloss = F.cross_entropy(pred, mask, reduction="mean")
            loss_dict[sub_area] = CEloss
        
        bg_loss = loss_dict['BG']
        et_loss = loss_dict['ET']
        tc_loss = loss_dict['TC']
        wt_loss = loss_dict['WT']
        custom_loss = self.w1 * et_loss + self.w2 * tc_loss + self.w3 * wt_loss + self.w_bg * bg_loss
        
        assert self.loss_type in ['custom', 'mean'], f'loss_type must be in ["custom", "mean"], but got {self.loss_type}'
        if self.loss_type == 'custom':
            global_loss = custom_loss
        else:
            global_loss = global_CEloss
        
        return global_loss, et_loss, tc_loss, wt_loss        


def splitClasses(y_pred, y_mask):
    """
    分割出每个类别的预测值和真实值
    :param y_pred: 预测值 [batch, 4, D, W, H]
    :param y_mask: 真实值 [batch, 4, D, W, H]
    """
    bg_pred  = y_pred[:, 0,...]
    ncr_pred = y_pred[:, 1,...] 
    ed_pred  = y_pred[:, 2,...] # 
    et_pred  = y_pred[:, 3,...] # 增强肿瘤

    bg_mask  = y_mask[:, 0,...]
    ncr_mask = y_mask[:, 1,...]
    ed_pred  = y_mask[:, 2,...] #
    et_pred  = y_mask[:, 3,...] # 增强肿瘤

    pred_list = [bg_pred, ncr_pred, ed_pred, et_pred]
    mask_list = [bg_mask, ncr_mask, ed_pred, et_pred]
    return pred_list, mask_list


def splitSubAreas(y_pred, y_mask):
    """
    分割出子区域
    :param y_pred: 预测值 [batch, 4, D, W, H]
    :param y_mask: 真实值 [batch, 4, D, W, H]
    """
    bg_pred = y_pred[:, 0,...]
    et_pred = y_pred[:, 3,...]
    tc_pred = y_pred[:, 1,...] + y_pred[:,3,...]
    wt_pred = y_pred[:, 1:,...].sum(dim=1)
    
    bg_mask = y_mask[:, 0,...]
    et_mask = y_mask[:, 3,...]
    tc_mask = y_mask[:, 1,...] + y_mask[:,3,...]
    wt_mask = y_mask[:, 1:,...].sum(dim=1)
    
    pred_list = [bg_pred, et_pred, tc_pred, wt_pred]
    mask_list = [bg_mask, et_mask, tc_mask, wt_mask]
    return pred_list, mask_list
    
def safe_loss(loss): # FIXME:当损失为nan时无法训练
    if loss is not None and not torch.isnan(loss):
        return loss
    else:
        return torch.nan_to_num(loss, nan=0.1)

if __name__ == '__main__':
    y_pred = torch.randn(2, 4, 128, 128, 128)
    # y_pred = torch.argmax(y_pred, dim=1)
    y_mask = torch.randint(0, 4, (2, 128, 128, 128))
    diceLossFunc = DiceLoss()
    ceLossFunc = CELoss()
    focallossFunc = FocalLoss()

    print(f"CELoss : {diceLossFunc(y_pred, y_mask)}")
    
    print(f"DiceLoss : {ceLossFunc(y_pred, y_mask)}")
    
    print(f"FocalLoss : {focallossFunc(y_pred, y_mask)}")
    