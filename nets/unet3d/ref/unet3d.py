# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/09/19 20:51:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UNet3D
=================================================
'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from modules import *
from torchsummary import summary

class UNet3D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0, use_bn:bool=True, use_ln:bool=False, use_dropout:bool=False, ln_spatial_shape:list=[]):
        super(UNet3D, self).__init__()     
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        self.encoder_use_list = (use_bn, use_ln, False, 0.1)
        self.decoder_use_list = (use_bn, use_ln, False, 0.1)
        # 编码器
        self.encoder1 = DoubleConv3x3(in_channels, 32, * self.encoder_use_list)
        self.encoder2 = DoubleConv3x3(32, 64, *self.encoder_use_list)
        self.encoder3 = DoubleConv3x3(64, 128, *self.encoder_use_list)
        self.encoder4 = DoubleConv3x3(128, 256, *self.encoder_use_list)
        self.encoder5 = DoubleConv3x3(256, 512, *self.encoder_use_list)

        # 解码器
        self.decoder1 = DoubleConv3x3(512, 256, *self.decoder_use_list)
        self.up1      = UpSampling2times(512, 256, *self.decoder_use_list)
        self.decoder2 = DoubleConv3x3(256, 128, *self.decoder_use_list)
        self.up2      = UpSampling2times(256, 128, *self.decoder_use_list)
        self.decoder3 = DoubleConv3x3(128, 64, *self.decoder_use_list)
        self.up3      = UpSampling2times(128, 64, *self.decoder_use_list)
        self.decoder4 = DoubleConv3x3(64, 32, *self.decoder_use_list)
        self.up4      = UpSampling2times(64, 32, *self.decoder_use_list)

        # 输出层
        self.output_conv = nn.Conv3d(32, out_channels, kernel_size=1)

        # 归一化层
        self.dropout = nn.Dropout3d(dropout_rate)
        
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # 编码器
        t1 = self.encoder1(x)                                                                   # [1, 32, 128, 128, 128]
        t2 = self.encoder2(F.max_pool3d(t1, 2, 2))                                              # [1, 64, 64, 64, 64] 
        t3 = self.encoder3(F.max_pool3d(t2, 2, 2))                                              # [1, 128, 32, 32, 32]
        t4 = self.encoder4(F.max_pool3d(t3, 2, 2))                                              # [1, 256, 16, 16, 16]
        out = self.encoder5(F.max_pool3d(t4, 2, 2))                                             # [1, 512, 8, 8, 8]

        # Dropout
        if self.use_dropout and (self.dropout_rate > 0):
            out = self.dropout(out)                                                              # [1, 512, 8, 8, 8]
        # 解码器        
        out = self.decoder1(torch.cat([self.up1(out), t4], dim=1))                               # [1, 256, 16, 16, 16]
        out = self.decoder2(torch.cat([self.up2(out), t3], dim=1))                               # [1, 128, 32, 32, 32]                      
        out = self.decoder3(torch.cat([self.up3(out), t2], dim=1))                               # [1, 64, 64, 64, 64]
        out = self.decoder4(torch.cat([self.up4(out), t1], dim=1))                               # [1, 32, 128, 128, 128]

        # 输出层
        out = self.output_conv(out)
        out = self.soft(out)

        return out


class UNet3d_dilation(UNet3D):
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0, use_bn:bool=True, use_ln:bool=False, use_dropout:bool=False, ln_spatial_shape:list=[]):
        super(UNet3d_dilation, self).__init__(in_channels, out_channels, dropout_rate, use_bn, use_ln, use_dropout, ln_spatial_shape)
        self.encoder1 = DoubleConvDilation(in_channels, 32, * self.encoder_use_list)
        self.encoder2 = DoubleConvDilation(32, 64, *self.encoder_use_list)
        self.encoder3 = DoubleConvDilation(64, 128, *self.encoder_use_list)
        self.encoder4 = DoubleConvDilation(128, 256, *self.encoder_use_list)
        self.encoder5 = DoubleConvDilation(256, 512, *self.encoder_use_list)

        self.decoder1 = DoubleConvDilation(512, 256, *self.decoder_use_list)
        self.up1      = UpSampling2times(512, 256, *self.decoder_use_list)

        self.decoder2 = DoubleConvDilation(256, 128, *self.decoder_use_list)
        self.up2      = UpSampling2times(256, 128, *self.decoder_use_list)

        self.decoder3 = DoubleConvDilation(128, 64, *self.decoder_use_list)
        self.up3      = UpSampling2times(128, 64, *self.decoder_use_list)

        self.decoder4 = DoubleConvDilation(64, 32, *self.decoder_use_list)
        self.up4      = UpSampling2times(64, 32, *self.decoder_use_list)
        # 输出层
        self.output_conv = nn.Conv3d(32, out_channels, kernel_size=1)

        # 归一化层
        self.dropout = nn.Dropout3d(dropout_rate)
        
        self.soft = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = super(UNet3d_dilation, self).forward(x)
        return out

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D_BN(in_channels=4, out_channels=4)
    print(model)
    input_tensor = torch.randn([1, 4, 128, 128, 128]).float()

    model.to(device)
    input_tensor = input_tensor.to(device)


    out = model(input_tensor)
    summary(model, (4, 128, 128, 128))
    print(out.shape)

    # def print_model_structure_with_shapes(model):
    #     for name, module in model.named_children():
    #         if isinstance(module, nn.Sequential):
    #             print(f"{name} (Sequential):")
    #             for sub_name, sub_module in module.named_children():
    #                 print(f"  {sub_name}: {sub_module}")
    #                 if hasattr(sub_module, 'weight'):
    #                     print(f"    Input shape: {sub_module.weight.shape}")
    #                     print(f"    Output shape: {sub_module.weight.shape}")
    #         else:
    #             print(f"{name}: {module}")
    #             if hasattr(module, 'weight'):
    #                 print(f"  Input shape: {module.weight.shape}")
    #                 print(f"  Output shape: {module.weight.shape}")
        # 使用示例
    # model = UNet3d_dilation(in_channels=4, out_channels=4)
    # print_model_structure_with_shapes(model)