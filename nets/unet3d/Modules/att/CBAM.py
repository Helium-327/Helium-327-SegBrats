# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/18 21:07:58
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: CBAM Module （通道注意力） 
=================================================
#!它通过在通道和空间两个维度上分别学习注意力权重，来增强特征图的表达能力。
# CBAM模块由两个主要部分组成：
    1. 通道注意力模块（Channel Attention Module）
    2. 空间注意力模块（Spatial Attention Module）
'''
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_dim, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_dim, in_dim // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_dim // ratio, in_dim, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avgout = self.fc(self.avg_pool(x))
        maxout = self.fc(self.max_pool(x))
        out = avgout + maxout
        return self.sigmoid(out)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    

class CBAM(nn.Module):
    def __init__(self, in_dim, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_dim, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        b, c, d, h, w = x.shape
        y = self.ca(x) * x
        z = self.sa(y) * y
        return z

if __name__ == '__main__':
    block = CBAM(32)  # 创建一个CBAM模块，输入通道为16
    input = torch.rand(1, 32, 128, 128, 128)  # 随机生成一个输入特征图
    output = block(input)  # 通过CBAM模块处理输入特征图
    print(output.shape)  # 打印输出的形状
        


        
        