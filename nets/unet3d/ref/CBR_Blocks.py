# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/09 21:10:38
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: Conv BN ReLU模块
=================================================
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
# from torchsummary import summary

class CBR_Block_3x3(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3, padding:int=1, dilation:int=1, stride:int=1):
        super(CBR_Block_3x3, self).__init__()
        # 参数
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out
    
class CBR_Block_5x5(CBR_Block_3x3):
    def __init__(self, in_channels:int, out_channels:int):
        super(CBR_Block_5x5, self).__init__(in_channels, out_channels)
        self.conv[0] = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2, dilation=1, bias=False)

class CBR_Block_Dilation(CBR_Block_3x3):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, padding:int, dilation:int):
        super(CBR_Block_Dilation, self).__init__(in_channels, out_channels)
        # 参数
        self.conv[0] = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)


class ResCBR_3x3(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super(ResCBR_3x3, self).__init__()
        self.conv = nn.Sequential(
            CBR_Block_3x3(in_channels, out_channels),
            CBR_Block_3x3(out_channels, out_channels)
        )
        del self.conv[1].conv[2]
        self.conv_1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)
    
    def forward(self, x):
        out_forward = self.conv(x)
        out_skip = self.conv_1x1(x)
        out = F.relu(out_forward + out_skip, inplace=True)
        return out

class ResCBR_dilation(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, padding:int, dilation:int):
        super(ResCBR_dilation, self).__init__()
        # 参数
        self.conv = nn.Sequential(
                    CBR_Block_Dilation(in_channels, out_channels, kernel_size, padding, dilation),
                    CBR_Block_Dilation(out_channels, out_channels, kernel_size, padding, dilation)
                )
        del self.conv[1].conv[2]
        self.conv_1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        out_forward = self.conv(x)
        out_skip = self.conv_1x1(x)
        out = F.relu(out_forward + out_skip, inplace=True)
        return out