'''
# -*- coding: UTF-8 -*-
================================================
*      CREATE ON: 2024/09/19 20:51:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: UNet3D
=================================================

'''


import torch
import numpy as np
import torch.nn as nn


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


class UNet_3d_22M_32(nn.Module):        # FIXME: 初始化之后损失异常
    def __init__(self, in_channels, out_channels):
        super(UNet_3d_22M_32, self).__init__()
        
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512)

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
        down2_out = self.down2(self.MaxPooling3d(down1_out))                    # 128 x 112 x 112 x 112
        down3_out = self.down3(self.MaxPooling3d(down2_out))                    # 256 x 56 x 56 x 56
        down4_out = self.down4(self.MaxPooling3d(down3_out))                    # 512 x 28 x 28 x 28
        down5_out = self.down5(self.MaxPooling3d(down4_out))                    # 1024 x 14 x 14 x 14

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
    def __init__(self, in_channels, out_channels):
        super(UNet_3d_22M_64, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

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
    def __init__(self, in_channels, out_channels):
        super(UNet_3d_90M, self).__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)

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
    def __init__(self, in_channels, num_classes):
        super(UNet_3d_48M, self).__init__()
        # self.ker_init = nn.init.he_normal_
        
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
        self.dropout = nn.Dropout(p=0.2)
        
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



"""---------------------------------------- 权重初始化 ----------------------------------------------"""
def init_weights_pro(model, init_type='normal', activation='relu', init_gain=0.02, always_init=True):
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

    sample = np.random.rand(16, 155, 155, 155)
    sample_3d = np.random.rand(1, 4, 144, 128, 128)
    label  = np.randint(0, 3, (1, 144, 128, 128))
    sample.shape
    model = UNet_3d_22M_32(in_channels=4, num_classes=3)
    # print(model)
    model.to(device)

    sample_tensor = torch.from_numpy(sample_3d).float()
    out = model(sample_tensor.to(device))
    print(out.shape)