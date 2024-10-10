# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/08 19:07:22
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 使用 dilation 为 2 的 膨胀卷积， 等效于 5x5的卷积
=================================================
参数：


'''


import torch
import torch.nn as nn
from torch.nn import functional as F
# from nets.unet3d.ref.modules import *
from ref.modules import *


class UNet3D_dilation(nn.Module):
    """
    膨胀卷积
    ================================================================
    Total params: 36,395,044
    Trainable params: 36,395,044
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 32.00
    Forward/backward pass size (MB): 13736.00
    Params size (MB): 138.84
    Estimated Total Size (MB): 13906.84
    ----------------------------------------------------------------
    """
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0, use_bn:bool=True, use_ln:bool=False, use_dropout:bool=False, ln_spatial_shape:list=[]):
        super(UNet3D_dilation, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
        # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
        # 编码器
        self.encoder1 = nn.Sequential(
            CBR_Block_Dilation(in_channels, 32, 3, 2, 2), 
            # CBR_Block_Dilation(32, 32),
        )
        self.encoder2 = nn.Sequential(
            CBR_Block_Dilation(32, 64, 3, 2, 2),
            # CBR_Block_Dilation(64, 64),
        )
        self.encoder3 = nn.Sequential(
            CBR_Block_Dilation(64, 128, 3, 2, 2),
            # CBR_Block_Dilation(128, 128),
        )
        self.encoder4 = nn.Sequential(
            CBR_Block_Dilation(128, 256, 3, 2, 2),
            # CBR_Block_Dilation(256, 256),
        )
        self.encoder5 = nn.Sequential(
            CBR_Block_Dilation(256, 512, 3, 2, 2),
            # CBR_Block_Dilation(512, 512),
        )

        # 解码器
        self.decoder1 = nn.Sequential(
            CBR_Block_Dilation(512, 256, 3, 2, 2),
            # CBR_Block_Dilation(256, 256),
        )
        self.up1      = nn.Sequential(
            Up_Block(512, 512, 4, 2, 1),
            CBR_Block_3x3(512, 256),
        )


        self.decoder2 = nn.Sequential(
            CBR_Block_Dilation(256, 128, 3, 2, 2),
            # CBR_Block_Dilation(128, 128),
        )
        self.up2      = nn.Sequential(
            Up_Block(256, 256, 4, 2, 1),
            CBR_Block_Dilation(256, 128, 3, 2, 2),
        )


        self.decoder3 = nn.Sequential(
            CBR_Block_Dilation(128, 64, 3, 2, 2),
            # CBR_Block_Dilation(64, 64),
        )
        self.up3      = nn.Sequential(
            Up_Block(128, 128, 4, 2, 1),
            CBR_Block_Dilation(128, 64, 3, 2, 2),
        )

        self.decoder4 = nn.Sequential(
            CBR_Block_Dilation(64, 32, 3, 2, 2),
            # CBR_Block_Dilation(32, 32),
        )
        self.up4      = nn.Sequential(
            Up_Block(64, 64, 4, 2, 1),
            CBR_Block_Dilation(64, 32, 3, 2, 2),
        )

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

class UNet3D_ResDilation(nn.Module):
    """基于 3D 残差卷积的 UNet 模型。
    ================================================================
    Total params: 51,053,348
    Trainable params: 51,053,348
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 32.00
    Forward/backward pass size (MB): 25988.00
    Params size (MB): 194.75
    Estimated Total Size (MB): 26214.75
    ----------------------------------------------------------------
    """
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0, use_bn:bool=True, use_ln:bool=False, use_dropout:bool=False, ln_spatial_shape:list=[]):
        super(UNet3D_ResDilation, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
        # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
        # 编码器
        self.encoder1 = nn.Sequential(
            ResCBR_dilation(in_channels, 32, 3, 2, 2), 
            # CBR_Block_Dilation(32, 32),
        )
        self.encoder2 = nn.Sequential(
            ResCBR_dilation(32, 64, 3, 2, 2),
            # CBR_Block_Dilation(64, 64),
        )
        self.encoder3 = nn.Sequential(
            ResCBR_dilation(64, 128, 3, 2, 2),
            # CBR_Block_Dilation(128, 128),
        )
        self.encoder4 = nn.Sequential(
            ResCBR_dilation(128, 256, 3, 2, 2),
            # CBR_Block_Dilation(256, 256),
        )
        self.encoder5 = nn.Sequential(
            ResCBR_dilation(256, 512, 3, 2, 2),
            # CBR_Block_Dilation(512, 512),
        )

        # 解码器
        self.decoder1 = nn.Sequential(
            ResCBR_dilation(512, 256, 3, 2, 2),
            # CBR_Block_Dilation(256, 256),
        )
        self.up1      = nn.Sequential(
            Up_Block(512, 512, 4, 2, 1),
            ResCBR_3x3(512, 256),
        )


        self.decoder2 = nn.Sequential(
            ResCBR_dilation(256, 128, 3, 2, 2),
            # CBR_Block_Dilation(128, 128),
        )
        self.up2      = nn.Sequential(
            Up_Block(256, 256, 4, 2, 1),
            ResCBR_3x3(256, 128),
        )


        self.decoder3 = nn.Sequential(
            ResCBR_dilation(128, 64, 3, 2, 2),
            # CBR_Block_Dilation(64, 64),
        )
        self.up3      = nn.Sequential(
            Up_Block(128, 128, 4, 2, 1),
            ResCBR_3x3(128, 64),
        )

        self.decoder4 = nn.Sequential(
            ResCBR_dilation(64, 32, 3, 2, 2),
            # CBR_Block_Dilation(32, 32),
        )
        self.up4      = nn.Sequential(
            Up_Block(64, 64, 4, 2, 1),
            ResCBR_3x3(64, 32),
        )

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
    
if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D_ResDilation(in_channels=4, out_channels=4)
    print(model)
    input_tensor = torch.randn([1, 4, 128, 128, 128]).float()

    model.to(device)
    input_tensor = input_tensor.to(device)
    summary(model, (4, 128, 128, 128))


    # out = model(input_tensor)
    # print(out.shape)