# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/21 20:57:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: unet3d的构建基础模块
=================================================
'''


import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.unet3d.Modules.Attentions3D import CBAM

class CBR_Block_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CBR_Block_3x3, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.down_sample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up_sample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                CBR_Block_3x3(in_channels, out_channels),
                CBR_Block_3x3(out_channels, out_channels)
                )
        else:
            self.up_sample = nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                CBR_Block_3x3(in_channels, out_channels)
                )
        
    def forward(self, x):
        return self.up_sample(x)
    
class ResAttCBR_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, att=False):
        super().__init__()
        block_list = []
        block_list.append(CBR_Block_3x3(in_channels, out_channels))
        if att:
            block_list.append(CBAM(out_channels))
        block_list.append(CBR_Block_3x3(out_channels, out_channels))
        self.double_cbr = nn.Sequential(*block_list)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return F.relu(self.conv(x) + self.double_cbr(x))


class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch5 = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.out_conv = nn.Conv3d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ], dim=1)
        out = F.relu(self.out_conv(out))
        return out