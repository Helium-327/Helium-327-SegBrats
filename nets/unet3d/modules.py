# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/09/27 19:34:42
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UNet3D的模块
=================================================
'''

import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class DoubleConv3x3(nn.Module):
    """双层3x3的卷积层
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param use_bn: 是否使用BN层
    :param use_ln: 是否使用LN层
    :param use_dropout: 是否使用Dropout层
    :param dropout_rate: Dropout层的概率
    :param ln_spatial_shape: LN层的空间形状
    """
    def __init__(self, in_channels:int, out_channels:int, use_bn=True, use_ln=False, use_dropout=False, dropout_rate=0, ln_spatial_shape:list=[]):
        super(DoubleConv3x3, self).__init__()
        # 参数
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.ln_spatial_shape = ln_spatial_shape

        # 卷积层
        if use_bn:
            self.conv3x3 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif use_ln:
            self.conv3x3 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LayerNorm([out_channels, *ln_spatial_shape]),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.LayerNorm([out_channels, *ln_spatial_shape]),
                nn.ReLU(inplace=True)
            )
        else:
            raise"Error: no normalization layer is used!"

        self.dropout = nn.Dropout3d(self.dropout_rate)

    def forward(self, x):
        out = self.conv3x3(x)
        if self.use_dropout:
            out = self.dropout(out)
        return out

class Conv5x5(nn.Module):
    """5x5的卷积层
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param use_bn: 是否使用BN层
    :param use_ln: 是否使用LN层
    :param use_dropout: 是否使用Dropout层
    :param dropout_rate: Dropout层的概率
    :param ln_spatial_shape: LN层的空间形状
    """
    def __init__(self, in_channels:int, out_channels:int, use_bn=True, use_ln=False, use_dropout=False, dropout_rate=0, ln_spatial_shape:list=[]):
        super(Conv5x5, self).__init__()
        # 参数
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.ln_spatial_shape = ln_spatial_shape

        # 卷积层
        if use_bn:
            self.conv3x3 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                # nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                # nn.BatchNorm3d(out_channels),
                # nn.ReLU(inplace=True)
            )
        elif use_ln:
            self.conv3x3 = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.LayerNorm([out_channels, *ln_spatial_shape]),
                nn.ReLU(inplace=True),
                # nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                # nn.LayerNorm([out_channels, *ln_spatial_shape]),
                # nn.ReLU(inplace=True)
            )
        else:
            raise"Error: no normalization layer is used!"

        self.dropout = nn.Dropout3d(self.dropout_rate)

    def forward(self, x):
        out = self.conv3x3(x)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class UpSampling2times(nn.Module):
    """上采样2倍的卷积层
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param use_bn: 是否使用BN层
    :param use_ln: 是否使用LN层
    :param use_dropout: 是否使用Dropout层
    :param dropout_rate: Dropout层的概率
    :param ln_spatial_shape: LN层的空间形状
    """
    def __init__(self, in_channels:int, out_channels:int, use_bn=True, use_ln=False, use_dropout=False, dropout_rate=0, ln_spatial_shape:list=[]):
        super(UpSampling2times, self).__init__()    
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.ln_spatial_shape = ln_spatial_shape

        if use_bn:
            self.UpSampling2times = nn.Sequential(
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif use_ln:
            self.UpSampling2times = nn.Sequential(
                nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2),
                nn.LayerNorm([in_channels, *(2*ln_spatial_shape)]),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LayerNorm([out_channels, *(2*ln_spatial_shape)]),
                nn.ReLU(inplace=True)
            )
        else:
            raise"Error: no normalization layer is used!"
        
        self.dropout = nn.Dropout3d(self.dropout_rate)

    def forward(self, x):
        out = self.UpSampling2times(x)
        if self.use_dropout:
            out = self.dropout(out)
        return out