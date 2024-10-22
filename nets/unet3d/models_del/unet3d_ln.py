# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/08 19:04:22
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 使用 Layer Norm的 UNet3D 
=================================================
参数:
================================================================
Total params: 939,721,796
Trainable params: 939,721,796
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 32.00
Forward/backward pass size (MB): 16624.00
Params size (MB): 3584.75
Estimated Total Size (MB): 20240.75
----------------------------------------------------------------
'''
import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.unet3d.ref.modules import *
from torchsummary import summary



class UNet3D_LN(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0, use_dropout:bool=False):
        super(UNet3D_LN, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        # 编码器
        self.encoder1 = nn.Sequential(
            CLR_Block_3x3(in_channels, 32, [128, 128, 128]),
            CLR_Block_3x3(32, 32, [128, 128, 128]),
        )
        self.encoder2 = nn.Sequential(
            CLR_Block_3x3(32, 64, [64, 64, 64]),
            CLR_Block_3x3(64, 64, [64, 64, 64]),
        )
        self.encoder3 = nn.Sequential(
            CLR_Block_3x3(64, 128, [32, 32, 32]),
            CLR_Block_3x3(128, 128, [32, 32, 32]),
        )
        self.encoder4 = nn.Sequential(
            CLR_Block_3x3(128, 256, [16, 16, 16]),
            CLR_Block_3x3(256, 256, [16, 16, 16]),
        )
        self.encoder5 = nn.Sequential(
            CLR_Block_3x3(256, 512, [8, 8, 8]),
            CLR_Block_3x3(512, 512, [8, 8, 8]),
        )

        # 解码器
        self.decoder1 = nn.Sequential(
            CLR_Block_3x3(512, 256, [16, 16, 16]),
            CLR_Block_3x3(256, 256, [16, 16, 16]),
        )
        self.up1      = nn.Sequential(
            Up_Block(512, 512, 4, 2, 1),
            CLR_Block_3x3(512, 256, [16, 16, 16]),
        )


        self.decoder2 = nn.Sequential(
            CLR_Block_3x3(256, 128, [32, 32, 32]),
            CLR_Block_3x3(128, 128, [32, 32, 32]),
        )
        self.up2      = nn.Sequential(
            Up_Block(256, 256, 4, 2, 1),
            CLR_Block_3x3(256, 128, [32, 32, 32]),
        )


        self.decoder3 = nn.Sequential(
            CLR_Block_3x3(128, 64, [64, 64, 64]),
            CLR_Block_3x3(64, 64, [64, 64, 64]),
        )
        self.up3      = nn.Sequential(
            Up_Block(128, 64, 4, 2, 1),
            CLR_Block_3x3(64, 64, [64, 64, 64]),
        )

        self.decoder4 = nn.Sequential(
            CLR_Block_3x3(64, 32, [128, 128, 128]),
            CLR_Block_3x3(32, 32, [128, 128, 128]),
        )
        self.up4      = nn.Sequential(
            Up_Block(64, 32, 4, 2, 1),
            CLR_Block_3x3(32, 32, [128, 128, 128]),
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
    model = UNet3D_LN(in_channels=4, out_channels=4)
    print(model)
    input_tensor = torch.randn([1, 4, 128, 128, 128]).float()

    model.to(device)
    input_tensor = input_tensor.to(device)


    out = model(input_tensor)
    summary(model, (4, 128, 128, 128))
    print(out.shape)