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
# from Attentions3D import *
# from Modules.Attentions3D import *
from nets.unet3d.Modules.Attentions3D import *

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
            # nn.ReLU(inplace=True),
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
            # nn.ReLU(inplace=True),
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

class D_Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
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
            # nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3), # 膨胀卷积，膨胀率为2
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

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential( 
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual_out = self.residual(x)
        shortcut_out = self.shortcut(x)
        out = self.relu(residual_out + shortcut_out)
        return self.downsample(out)
    

class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.residual = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            CBAM(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skipped=None):
        x = self.upsample(x)
        if skipped is not None:
            x = torch.cat([x, skipped], dim=1)
        residual_out = self.residual(x)
        shortcut_out = self.shortcut(x)
        out = self.relu(residual_out + shortcut_out)
        return out
    
class FusionMagic(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2): # 32, 128
        super().__init__()
        # 分别对后两层的输入进行平均池化操作，得到每个通道的平均值
        self.avgpooloing = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm([in_channels, 1, 1, 1])
        self.layer_norm2 = nn.LayerNorm([in_channels*2, 1, 1, 1])

        # 使用SE_Block进行将cat之后的特征进行压缩激发
        # self.SE_layer1 = SE_Block(in_channels*) # in_channels*6 = in)_channels*2 + in_channels*2*2
        self.layer_norm3 = nn.LayerNorm([in_channels*4, 1, 1, 1])

        self.layer_norm4 = nn.LayerNorm([in_channels*6, 1, 1, 1])

        self.layer_norm5 = nn.LayerNorm([in_channels*7, 1, 1, 1])


        self.MLP = nn.Sequential(
            nn.Conv3d(in_channels=in_channels*7, out_channels=in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels*7, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.Conv1 = nn.Conv3d(in_channels*7, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs:list[torch.tensor]):
        # 对后两层的输入进行平均池化操作，得到每个通道的平均值
        x1 = self.avgpooloing(inputs[-1])
        # x1 = self.SE_layer1(x1)
        x1 = self.layer_norm1(x1)

        x2 = self.avgpooloing(inputs[-2])
        x2 = self.layer_norm2(x2)

        x3 = self.avgpooloing(inputs[0])
        x3 = self.layer_norm3(x3)

        out = torch.cat([x2, x3], dim=1)
        out = self.layer_norm4(out)

        out = torch.cat([x1, out], dim=1)
        out = self.layer_norm5(out)

        out = self.dropout(out)

        out = self.MLP(out)
        out = self.Conv1(out)
        # out = self.SE_layer1(out)

        out = self.sigmoid(out)

        return out
    
if __name__ == '__main__':
    from Modules.Attentions3D import *
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionMagic(32, 128).to(device)
    inputs = [torch.randn(1, 32, 128, 128, 128).to(device), torch.randn(1, 64, 64, 64, 64).to(device), torch.randn(1, 128, 32, 32, 32).to(device)]
    # print(model)
    output = model(inputs)
    print(output.shape)


        


        
        
        