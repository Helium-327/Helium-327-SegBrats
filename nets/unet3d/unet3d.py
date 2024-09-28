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
from .modules import *
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


    





    


"""---------------------------------------- 权重初始化 ----------------------------------------------"""
def init_weights_pro(model, init_type='normal', activation='relu', init_gain=0.02, always_init=False):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            if init_type == 'kaiming_normal':
                if isinstance(m, nn.ConvTranspose3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation)  # `fan_in` for ConvTranspose3d
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)  # `fan_out` for Conv3d
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.normal_(m.weight, 0, init_gain)
            
            if always_init or not (init_type in ['kaiming_normal', 'xavier_normal']):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm3d):
            if always_init or init_type in ['kaiming_normal', 'xavier_normal']:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            if init_type == 'kaiming_normal':
                if isinstance(m, nn.ConvTranspose3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.normal_(m.weight, 0, init_gain)
                
            if always_init:
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def init_weights_light(model,  init_gain=0.02):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, init_gain)
            nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=4, out_channels=4)
    input_tensor = torch.randn([1, 4, 128, 128, 128]).float()

    model.to(device)
    input_tensor = input_tensor.to(device)


    out = model(input_tensor)
    print(out.shape)
    summary(model, (4, 128, 128, 128))