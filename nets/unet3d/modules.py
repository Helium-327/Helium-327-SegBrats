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

class CBR_Block_3x3(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, padding:int=1, dilation:int=1, stride:int=1):
        super(CBR_Block_3x3, self).__init__()
        # 参数
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    
class CBR_Block_5x5(CBR_Block_3x3):
    def __init__(self, in_channels:int, out_channels:int):
        super(CBR_Block_5x5, self).__init__(in_channels, out_channels)
        self.conv[0] = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2, dilation=1, bias=True)

class CBR_Block_Dilation(CBR_Block_3x3):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, padding:int, dilation:int):
        super(CBR_Block_Dilation, self).__init__(in_channels, out_channels)
        # 参数
        self.conv[0] = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=True)



class CLR_Block_3x3(CBR_Block_3x3):
    def __init__(self, in_channels:int, out_channels:int, ln_spatial_shape:list=[]):
        super(CLR_Block_3x3, self).__init__(in_channels, out_channels)
        # 参数
        self.conv[2] = nn.LayerNorm([out_channels, *ln_spatial_shape])
    
    def forward(self, x):
        out = self.conv(x)
        return out

class CLR_Block_5x5(CBR_Block_5x5):
    def __init__(self, in_channels:int, out_channels:int, ln_spatial_shape:list=[]):
        super(CLR_Block_5x5, self).__init__(in_channels, out_channels)
        # 参数
        self.conv[2] = nn.LayerNorm([out_channels, *ln_spatial_shape])

class CLR_Block_Dilation(CBR_Block_Dilation):
    def __init__(self, in_channels:int, out_channels:int, ln_spatial_shape:list=[]):
        super(CLR_Block_Dilation, self).__init__(in_channels, out_channels)
        # 参数
        self.conv[2] = nn.LayerNorm([out_channels, *ln_spatial_shape])

class Up_Block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size, stride, padding):
        super(Up_Block, self).__init__()
        # 参数
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.up(x)
        return out
