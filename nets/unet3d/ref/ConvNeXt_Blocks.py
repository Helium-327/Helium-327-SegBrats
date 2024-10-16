# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/15 14:01:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: ConvNeXt模块
=================================================
'''
import torch 
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class ConvNextBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(ConvNextBlock, self).__init__()
        # self.d_conv5x5 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.d_conv7x7 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=3, dilation=3, bias=False)
        self.conv1x1_1 = nn.Conv3d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, bias=False)
        self.conv1x1_2 = nn.Conv3d(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(in_channels)
    def forward(self, x):
        res_out = x
        out = self.d_conv7x7(x)
        out = F.layer_norm(out, out.shape[1:])
        # out = self.bn(out)
        out = self.conv1x1_1(out)
        out = F.gelu(out)
        out = self.conv1x1_2(out)
        return out + res_out
        

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_data = torch.randn(1, 32, 128, 128, 128).to(device=device)

    model = ConvNextBlock(32, 64).to(device=device)
    output_data = model(input_data)

    print(output_data.shape)
    summary(model, input_size=(32, 128, 128, 128))
    print(model)