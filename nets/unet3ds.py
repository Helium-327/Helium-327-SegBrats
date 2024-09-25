# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/09/19 20:51:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UNet3D
=================================================
'''


import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class _make_conv_layer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, use_bn=True, use_ln=False, use_dropout=False, dropout_rate=0, ln_spatial_shape:list=[]):
        super(_make_conv_layer, self).__init__()
        # 参数
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.ln_spatial_shape = ln_spatial_shape

        # 卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.ln1 = nn.LayerNorm([out_channels, *ln_spatial_shape]) # 解包
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.ln2 = nn.LayerNorm([out_channels, *ln_spatial_shape])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(self.dropout_rate)

    def forward(self, x):
        if self.use_bn:
            out = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))
        elif self.use_ln:
            out = self.relu(self.ln2(self.conv2(self.relu(self.ln1(self.conv1(x))))))
        else:
           raise"Error: no normalization layer is used!"
        if self.use_dropout:
            out = self.dropout(out)
        return out
    
class UNet3D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0, use_bn:bool=True, use_ln:bool=False, use_dropout:bool=False, ln_spatial_shape:list=[]):
        super(UNet3D, self).__init__()     
        self.dropout_rate = dropout_rate
        self.encoder_use_list = (use_bn, use_ln, use_dropout, dropout_rate)
        self.decoder_use_list = (use_bn, use_ln, False, dropout_rate)
        # 编码器
        self.encoder1 = _make_conv_layer(in_channels, 32, * self.encoder_use_list)
        self.encoder2 = _make_conv_layer(32, 64, * self.encoder_use_list)
        self.encoder3 = _make_conv_layer(64, 128, * self.encoder_use_list)
        self.encoder4 = _make_conv_layer(128, 256, * self.encoder_use_list)
        self.encoder5 = _make_conv_layer(256, 512, * self.encoder_use_list)

        # 解码器
        self.decoder1 = _make_conv_layer(512, 256, *self.decoder_use_list)
        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = _make_conv_layer(256, 128, *self.decoder_use_list)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = _make_conv_layer(128, 64, *self.decoder_use_list)
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = _make_conv_layer(64, 32, *self.decoder_use_list)
        self.up4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)

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
        if self.dropout_rate > 0:
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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class UNet_3d_22M_32(nn.Module):   
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(UNet_3d_22M_32, self).__init__()
        
        self.dropout_p = dropout_p
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512)

        self.dropout = nn.Dropout3d(p=self.dropout_p)

        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.up1_conv = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up2_conv = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up3_conv = DoubleConv(128, 64)
        
        self.up4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up4_conv = DoubleConv(64, 32)

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

        self.MaxPooling3d = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.softmax = nn.Softmax(dim=1)
        
        # self.initialize_weights(init_type="kaiming_normal", activation="relu")

    def forward(self, x):
        down1_out = self.down1(x)                                               # 64 x 224 x 224 x 224
        down2_out = self.down2(F.max_pool3d(down1_out, 2, 2))                    # 128 x 112 x 112 x 112
        down3_out = self.down3(F.max_pool3d(down2_out, 2, 2))                    # 256 x 56 x 56 x 56
        down4_out = self.down4(F.max_pool3d(down3_out, 2, 2))                    # 512 x 28 x 28 x 28
        down5_out = self.down5(F.max_pool3d(down4_out, 2, 2))                    # 1024 x 14 x 14 x 14

        if self.dropout_p > 0:
            down5_out = self.dropout(down5_out)                                     # 1024 x 14 x 14 x 14

        up1_out = self.up1(down5_out)                                           # 512 x 28 x 28 x 28
        up1_cat_out = torch.cat([up1_out, down4_out], dim=1)                    # 1024 x 28 x 28 x 28
        up1_conv_out = self.up1_conv(up1_cat_out)                               # 512 x 28 x 28 x 28

        up2_out = self.up2(up1_conv_out)                                           # 256 x 56 x 56 x 56
        up2_cat_out = torch.cat([up2_out, down3_out], dim=1)                    # 512 x 56 x 56 x 56
        up2_conv_out = self.up2_conv(up2_cat_out)                               # 256 x 56 x 56 x 56

        up3_out = self.up3(up2_conv_out)                                           # 128 x 112 x 112 x 112
        up3_cat_out = torch.cat([up3_out, down2_out], dim=1)                    # 256 x 112 x 112 x 112
        up3_conv_out = self.up3_conv(up3_cat_out)                               # 128 x 112 x 112 x 112

        up4_out = self.up4(up3_conv_out)                                           # 64 x 224 x 224 x 224
        up4_cat_out = torch.cat([up4_out, down1_out], dim=1)                    # 128 x 224 x 224 x 224
        up4_conv_out = self.up4_conv(up4_cat_out)                               # 64 x 224 x 224 x 224

        out = self.out_conv(up4_conv_out)                                       # out_channel x 224 x 224 x 224
        
        out = self.softmax(out)
        return out
    

class UNet_3d_22M_64(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(UNet_3d_22M_64, self).__init__()
        self.dropout_p = dropout_p
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.dropout = nn.Dropout3d(p=self.dropout_p)

        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.up1_conv = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up2_conv = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up3_conv = DoubleConv(128, 64)

        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)

        self.MaxPooling3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)
        # self.initialize_weights(init_type="kaiming_normal", activation="relu")

    def forward(self, x):
        down1_out = self.down1(x)                                               # 64 x 224 x 224 x 224
        down2_out = self.down2(self.MaxPooling3d(down1_out))                    # 128 x 112 x 112 x 112
        down3_out = self.down3(self.MaxPooling3d(down2_out))                    # 256 x 56 x 56 x 56
        down4_out = self.down4(self.MaxPooling3d(down3_out))                    # 512 x 28 x 28 x 28

        if self.dropout_p > 0:
            down4_out = self.dropout(down4_out)

        up1_out = self.up1(down4_out)                                           # 256 x 56 x 56 x 56
        up1_cat_out = torch.cat([up1_out, down3_out], dim=1)                    # 512 x 56 x 56 x 56
        up1_conv_out = self.up1_conv(up1_cat_out)                                   # 256 x 56 x 56 x 56

        up2_out = self.up2(up1_conv_out)                                        # 128 x 112 x 112 x 112
        up2_cat_out = torch.cat([up2_out, down2_out], dim=1)                    # 256 x 112 x 112 x 112
        up2_conv_out = self.up2_conv(up2_cat_out)                               # 128 x 112 x 112 x 112

        up3_out = self.up3(up2_conv_out)                                        # 64 x 224 x 224 x 224
        up3_cat_out = torch.cat([up3_out, down1_out], dim=1)                    # 128 x 224 x 224 x 224
        up3_conv_out = self.up3_conv(up3_cat_out)                               # 64 x 224 x 224 x 224

        out = self.out_conv(up3_conv_out)                                       # out_channel x 224 x 224 x 224
        out = self.softmax(out)
        return out
    
class UNet_3d_90M(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(UNet_3d_90M, self).__init__()
        self.dropout_p = dropout_p

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.up1_conv = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.up2_conv = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up3_conv = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up4_conv = DoubleConv(128, 64)

        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)

        self.MaxPooling3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)

        # self.initialize_weights()

    def forward(self, x):
        down1_out = self.down1(x)                                               # 64 x 224 x 224 x 224
        down2_out = self.down2(self.MaxPooling3d(down1_out))                    # 128 x 112 x 112 x 112
        down3_out = self.down3(self.MaxPooling3d(down2_out))                    # 256 x 56 x 56 x 56
        down4_out = self.down4(self.MaxPooling3d(down3_out))                    # 512 x 28 x 28 x 28
        down5_out = self.down5(self.MaxPooling3d(down4_out))                    # 1024 x 14 x 14 x 14

        if self.dropout_p > 0:
            down5_out = self.dropout(down5_out)

        up1_out = self.up1(down5_out)                                           # 512 x 28 x 28 x 28
        up1_cat_out = torch.cat([up1_out, down4_out], dim=1)                    # 1024 x 28 x 28 x 28
        up1_conv_out = self.up1_conv(up1_cat_out)                               # 512 x 28 x 28 x 28


        up2_out = self.up2(up1_conv_out)                                           # 256 x 56 x 56 x 56
        up2_cat_out = torch.cat([up2_out, down3_out], dim=1)                    # 512 x 56 x 56 x 56
        up2_conv_out = self.up2_conv(up2_cat_out)                               # 256 x 56 x 56 x 56

        up3_out = self.up3(up2_conv_out)                                           # 128 x 112 x 112 x 112
        up3_cat_out = torch.cat([up3_out, down2_out], dim=1)                    # 256 x 112 x 112 x 112
        up3_conv_out = self.up3_conv(up3_cat_out)                               # 128 x 112 x 112 x 112

        up4_out = self.up4(up3_conv_out)                                           # 64 x 224 x 224 x 224
        up4_cat_out = torch.cat([up4_out, down1_out], dim=1)                    # 128 x 224 x 224 x 224
        up4_conv_out = self.up4_conv(up4_cat_out)                               # 64 x 224 x 224 x 224

        

        out = self.out_conv(up4_conv_out)                                       # out_channel x 224 x 224 x 224
        
        out = self.softmax(out)
        return out
    
class UNet_3d_48M(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_p=0):
        super(UNet_3d_48M, self).__init__()
        # self.ker_init = nn.init.he_normal_
        self.dropout_p = dropout_p
        self.maxPooling = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.Conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        ) # 
        self.Conv5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        ) # c = 512
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        # TODO: 转置卷积怎么用
        # H_out = (H_in - 1) * stride - 2 * padding + kernel_size
        self.upSampling3d_1 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=4, stride=2, padding=1), # 上采样 8 ---> 16
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 256, kernel_size=3, padding=1), 
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        ) # c =256
        
        # 与Conv4的输出concate拼接，之后的 c= 512
        self.Conv6 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.upSampling3d_2  = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1), # 上采样 16 ---> 32
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        # 与Conv3的输出concate拼接，之后的 c= 256
        self.Conv7 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.upSampling3d_3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1), # 上采样 32 ---> 64
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        # 与Conv2的输出concate拼接，之后的 c= 128
        
        self.Conv8 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upSampling3d_4 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1), # 上采样 64 ---> 128
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        # 与Conv1的输出concate拼接，之后的 c= 64
        self.Conv9 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.ConvOutput = nn.Conv3d(32, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.initialize_weights(init_type='kaiming_normal')
        
        
    def forward(self, x):
        input_layer = self.Conv1(x)  # 2 x 128 x 128 ----> 32 x 128 x 128
        down1 = self.maxPooling(input_layer) # 32 x 64 x 64
        down2 = self.maxPooling(self.Conv2(down1)) # 64 x 32 x 32
        down3 = self.maxPooling(self.Conv3(down2)) # 128 x 16 x 16
        down4 = self.maxPooling(self.Conv4(down3)) # 256 x 8 x 8
        down_ouput = self.Conv5(down4) # 512 x 8 x 8
        
        if self.dropout_p > 0:
            dropout_output = self.dropout(down_ouput) # 512 x 8 x 8
        up1 = self.upSampling3d_1(dropout_output) # 256 x 16 x 16
        up1_cat_down4 = torch.cat([up1, self.Conv4(down3)], dim=1) # [256 x 16 x 16, 256 x 16 x 16] ----> 512 x 16 x 16
        up2 = self.Conv6(up1_cat_down4) # 256 x 32 x 32
        up3 = self.upSampling3d_2(up2) # 128 x 32 x 32
        up3_cat_down3 = torch.cat([up3, self.Conv3(down2)], dim=1) # [128 x 32 x 32, 128 x 32 x 32] ----> 256 x 32 x 32
        up4 = self.Conv7(up3_cat_down3) # 128 x 32 x 32
        up5 = self.upSampling3d_3(up4)  # 64 x 64 x 64
        up5_cat_down2 = torch.cat([up5, self.Conv2(down1)], dim=1) # [64 x 64 x 64, 64 x 64 x 64] ----> 128 x 64 x 64
        up6 = self.Conv8(up5_cat_down2) # 64 x 64 x 64
        up7 = self.upSampling3d_4(up6)  # 32 x 128 x 128
        up7_cat_down1 = torch.cat([up7, input_layer], dim=1) # [32 x 128 x 128, 32 x 128 x 128] ----> 64 x 128 x 128
        up8 = self.Conv9(up7_cat_down1) # 32 x 128 x 128
        output = self.ConvOutput(up8) # num_class x 128 x 128
        out = self.softmax(output)
        return out     

class UNet3d_bn_256(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(UNet3d_bn_256, self).__init__()
        self.dropout_p = dropout_p
        self.encoder1 = DoubleConv(in_channels, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder4 = DoubleConv(128, 256)
        # self.encoder5 = DoubleConv(256, 512) 
        self.dropout = nn.Dropout(p=self.dropout_p)
        # self.decoder1 = DoubleConv(512, 256)
        # self.con_trans1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(256, 128)
        self.conv_trans1 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(128, 64)
        self.conv_trans2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(64, 32)
        self.conv_trans3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.out_conv = DoubleConv(32, out_channels)

        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # 编码器部分
        t1 = self.encoder1(x)                                               # 32 x 128 x 128 x 128
        out = F.max_pool3d(t1, 2, 2)                                        # 32 x 64 x 64 x 64
                                    
        t2 = self.encoder2(out)                                             # 64 x 64 x 64 x 64
        out = F.max_pool3d(t2, 2, 2)                                        # 64 x 32 x 32 x 32
        
        t3 = self.encoder3(out)                                             # 128 x 32 x 32 x 32
        out = F.max_pool3d(t3, 2, 2)                                        # 128 x 16 x 16 x 16
        
        out = self.encoder4(out)                                            # 256 x 16 x 16 x 16
        
        if self.dropout_p > 0:
            out = self.dropout(out)                                         # 256 x 16 x 16 x 16
        # 解码器部分
        out = self.conv_trans1(out)                                         # 128 x 32 x 32 x 32
        out = self.decoder1(torch.cat([out, t3], dim=1))                    # 128 x 32 x 32 x 32
        
        out = self.conv_trans2(out)                                         # 64 x 64 x 64 x 64
        out = self.decoder2(torch.cat([out, t2], dim=1))                    # 64 x 64 x 64 x 64                

        out = self.conv_trans3(out)                                         # 32 x 128 x 128 x 128
        out = self.decoder3(torch.cat([out, t1], dim=1))                    # 32 x 128 x 128 x 128

        out = self.out_conv(out)                                            # out_channels x 128 x 128
        
        out = self.softmax(out)                                             # softmax
        return out


class UNet3d_bn_512(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(UNet3d_bn_512, self).__init__()
        self.dropout_p = dropout_p
        self.encoder1 = DoubleConv(in_channels, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder4 = DoubleConv(128, 256)
        self.encoder5 = DoubleConv(256, 512) 

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.decoder1 = DoubleConv(512, 256)
        self.conv_trans1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.conv_trans2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)
        self.conv_trans3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(64, 32)
        self.conv_trans4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.out_conv = DoubleConv(32, out_channels)

        self.soft = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # 编码器部分
        t1 = self.encoder1(x)                                               # 32 x 128 x 128 x 128
        out = F.max_pool3d(t1, 2, 2)                                        # 32 x 64 x 64 x 64
                                    
        t2 = self.encoder2(out)                                             # 64 x 64 x 64 x 64
        out = F.max_pool3d(t2, 2, 2)                                        # 64 x 32 x 32 x 32
        
        t3 = self.encoder3(out)                                             # 128 x 32 x 32 x 32
        out = F.max_pool3d(t3, 2, 2)                                        # 128 x 16 x 16 x 16
        
        t4 = self.encoder4(out)                                             # 256 x 16 x 16 x 16
        out = F.max_pool3d(t4, 2, 2)                                        # 256 x 8 x 8 x 8
        
        out = self.encoder5(out)                                            # 512 x 8 x 8 x 8
        
        if self.dropout_p > 0:
            out = self.dropout(out)                                          # 512 x 8 x 8 x 8
        
        out = self.conv_trans1(out)                                         # 256 x 16 x 16 x 16
        out = self.decoder1(torch.cat([out, t4], dim=1))                    # 256 x 16 x 16 x 16
        
        out = self.conv_trans2(out)                                          # 128 x 32 x 32 x 32
        out = self.decoder2(torch.cat([out, t3], dim=1))                    # 128 x 32 x 32 x 32
        
        out = self.conv_trans3(out)                                         # 64 x 64 x 64 x 64
        out = self.decoder3(torch.cat([out, t2], dim=1))                    # 64 x 64 x 64 x 64                

        out = self.conv_trans4(out)                                         # 32 x 128 x 128 x 128
        out = self.decoder4(torch.cat([out, t1], dim=1))                    # 32 x 128 x 128 x 128

        out = self.out_conv(out)                                            # out_channels x 128 x 128
        
        out = self.soft(out)                                             # softmax
        return out

# simple UNet3d_ln
class UNet_3d_ln(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(UNet_3d_ln, self).__init__()
        self.dropout_p = dropout_p
        self.encoder1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.encoder5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.decoder1 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.conv_trans1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.conv_trans2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv_trans3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv_trans4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)

        self.soft = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # 编码器
        out = self.encoder1(x)                                              # 32 x 128 x 128 x 128
        out = F.relu(F.layer_norm(out, out.shape[-3:]))                     
        t1 = out                                                            # 32 x 128 x 128 x 128
        
        out = F.max_pool3d(t1, 2, 2)                                        # 32 x 64 x 64 x 64
        out = self.encoder2(out)                                            # 64 x 64 x 64 x 64
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        t2 = out                                                            # 64 x 64 x 64 x 64

        out = F.max_pool3d(t2, 2, 2)                                        # 64 x 32 x 32 x 32
        out = self.encoder3(out)                                            # 128 x 32 x 32 x 32
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        t3 = out                                                            # 128 x 32 x 32 x 32

        out = F.max_pool3d(t3, 2, 2)                                        # 128 x 16 x 16 x 16
        out = self.encoder4(out)                                            # 256 x 16 x 16 x 16
        out = F.relu(F.layer_norm(out, out.shape[-3:]))                     
        t4 = out                                                            # 256 x 16 x 16 x 16
        
        out = F.max_pool3d(t4, 2, 2)                                        # 256 x 8 x 8 x 8
        out = self.encoder5(out)                                            # 512 x 8 x 8 x 8
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        
        if self.dropout_p > 0:
            out = self.dropout(out)

        # 解码器
        out = self.conv_trans1(out)                                         # 256 x 16 x 16 x 16
        out = self.decoder1(torch.cat([out, t4], dim=1))                    # 256 x 16 x 16 x 16
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        
        out = self.conv_trans2(out)                                         # 128 x 32 x 32 x 32
        out = self.decoder2(torch.cat([out, t3], dim=1))                    # 128 x 32 x 32 x 32
        out = F.relu(F.layer_norm(out, out.shape[-3:]))

        out = self.conv_trans3(out)                                         # 64 x 64 x 64 x 64
        out = self.decoder3(torch.cat([out, t2], dim=1))                    # 64 x 64 x 64 x 64
        out = F.relu(F.layer_norm(out, out.shape[-3:]))                 
        
        out = self.conv_trans4(out)                                         # 32 x 128 x 128 x 128
        out = self.decoder4(torch.cat([out, t1], dim=1))                    # 32 x 128 x 128 x 128
        out = F.relu(F.layer_norm(out, out.shape[-3:]))                     
        
        out = self.out_conv(out)                                            # out_channels x 128 x 128 x 128
        
        out = self.soft(out)                                                # softmax
        
        return out


# 改进
class UNet_3d_ln2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(UNet_3d_ln2, self).__init__()
        self.dropout_p = dropout_p
        self.encoder1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.encoder5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)

        self.dropout = nn.Dropout3d(p=self.dropout_p)
        self.conv_32    = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv_64    = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.conv_128    = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv_256    = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.conv_512    = nn.Conv3d(512, 512, kernel_size=3, padding=1)    
        
        self.decoder1 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.conv_trans1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.conv_trans2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv_trans3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv_trans4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)

        self.soft = nn.Softmax(dim=1)
        
        
        
    def forward(self, x):
        # 编码器
        out = self.encoder1(x)                                              # 32 x 128 x 128 x 128
        out = F.relu(F.layer_norm(out, out.shape[-3:]))                     
        out = self.conv_32(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        t1 = out                                                            # 32 x 128 x 128 x 128
        
        out = F.max_pool3d(t1, 2, 2)                                        # 32 x 64 x 64 x 64
        out = self.encoder2(out)                                            # 64 x 64 x 64 x 64
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        out = self.conv_64(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        t2 = out                                                            # 64 x 64 x 64 x 64

        out = F.max_pool3d(t2, 2, 2)                                        # 64 x 32 x 32 x 32
        out = self.encoder3(out)                                            # 128 x 32 x 32 x 32
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        out = self.conv_128(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        t3 = out                                                            # 128 x 32 x 32 x 32

        out = F.max_pool3d(t3, 2, 2)                                        # 128 x 16 x 16 x 16
        out = self.encoder4(out)                                            # 256 x 16 x 16 x 16
        out = F.relu(F.layer_norm(out, out.shape[-3:]))                     
        out = self.conv_256(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        t4 = out                                                            # 256 x 16 x 16 x 16
        
        out = F.max_pool3d(t4, 2, 2)                                        # 256 x 8 x 8 x 8
        out = self.encoder5(out)                                            # 512 x 8 x 8 x 8
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        out = self.conv_512(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        
        if self.dropout_p > 0:
            out = self.dropout(out)                                          # 256 x 16 x 16 x 16

        # 解码器
        out = self.conv_trans1(out)                                         # 256 x 16 x 16 x 16
        out = self.decoder1(torch.cat([out, t4], dim=1))                    # 256 x 16 x 16 x 16
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        out = self.conv_256(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        
        out = self.conv_trans2(out)                                         # 128 x 32 x 32 x 32
        out = self.decoder2(torch.cat([out, t3], dim=1))                    # 128 x 32 x 32 x 32
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        out = self.conv_128(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))

        out = self.conv_trans3(out)                                         # 64 x 64 x 64 x 64
        out = self.decoder3(torch.cat([out, t2], dim=1))                    # 64 x 64 x 64 x 64
        out = F.relu(F.layer_norm(out, out.shape[-3:]))                 
        out = self.conv_64(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        
        out = self.conv_trans4(out)                                         # 32 x 128 x 128 x 128
        out = self.decoder4(torch.cat([out, t1], dim=1))                    # 32 x 128 x 128 x 128
        out = F.relu(F.layer_norm(out, out.shape[-3:]))                     
        out = self.conv_32(out)
        out = F.relu(F.layer_norm(out, out.shape[-3:]))
        
        out = self.out_conv(out)                                            # out_channels x 128 x 128 x 128
        
        out = self.soft(out)                                                # softmax
        
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
    model = UNet_3d_ln2(in_channels=4, out_channels=4)
    input_tensor = torch.randn([1, 4, 128, 128, 128]).float()

    model.to(device)
    input_tensor = input_tensor.to(device)


    out = model(input_tensor)
    print(out.shape)
    summary(model, (4, 128, 128, 128))