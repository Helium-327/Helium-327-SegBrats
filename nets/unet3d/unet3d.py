# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/19 14:36:30
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION:  构建通用的UNET3D模型
=================================================
'''
import torch
import torch.nn as nn
from nets.unet3d.Modules.Attentions3D import *
from nets.unet3d.Modules.UnetBlocks3D import *


from torchinfo import summary
from torch.nn import functional as F
from torchcrf import CRF

'''========================================== 原UNET3D网络 ============================================'''
class UNET3D(nn.Module):
    """原UNET3D网络结构:
    features=[32, 64, 128, 256] 时最合适，
    当features=[64, 128, 256]时，占用显存会比较大，
    当features=[16, 32, 64, 128, 256]时，预测可视化效果不好
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 features=[32, 64, 128, 256],
                 down_att =False,
                 up_att =False,
                 bottom_att = False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.down_att    = down_att
        self.up_att      = up_att
        self.bottom_att  = bottom_att
        self.encoders_features  = features
        self.decoders_features  = (features + [features[-1]*2])[::-1]

        self.encoders     = nn.ModuleList()
        self.decoders     = nn.ModuleList()
        self._make_encoders()

        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(features[-1], features[-1]*2),
            CBR_Block_3x3(features[-1]*2, features[-1]*2)
        )
        self._make_decoders()
        self.out_conv = nn.Conv3d(self.decoders_features[-1], self.out_channels, kernel_size=1)
        self.soft_max = nn.Softmax(dim=1)
        # self.crf      = CRF(out_channels)

    def _make_encoders(self):
        for i in range(len(self.encoders_features)):
            if i == 0:
                self.encoders.append(CBR_Block_3x3(self.in_channels, self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            else:
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i-1], self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            self.encoders.append(DownSample())
            
    def _make_decoders(self):
        for i in range(len(self.decoders_features)):
            if i == len(self.decoders_features)-1:
                continue
            else:
                self.decoders.append(UpSample(self.decoders_features[i], self.decoders_features[i+1], trilinear=False))
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i], self.decoders_features[i+1]))
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i+1], self.decoders_features[i+1]))
            
    def forward(self, x):
        out = x
        # print(f'Input shape: {out.shape}')
        skip_out = []
        for m in self.encoders:
            if isinstance(m, DownSample):
                skip_out.append(out)
                print(f"skip_out: {out.shape}")
            out = m(out)
            print(f'Encoder shape: {out.shape}')
            print("-" * 50)

        for t in skip_out:
            print(f'Skip connection shape: {t.shape}')

        out = self.bottom_layer(out)

        for m in self.decoders:
            if isinstance(m, UpSample):
                out = m(out)
                print(f'up shape : {out.shape}')
                print(f'skip shape : {skip_out[-1].shape}')
                out = torch.cat([out, skip_out.pop()], dim=1)
                print(f'after cat shape : {out.shape}')
                print("-" * 50)
            else:
                out = m(out)
                print(f'Decoder shape: {out.shape}')
        out = self.out_conv(out)
        out = self.soft_max(out)
        # out = self.crf
        return out

class UNET3D_v2(nn.Module):
    """使用先上采样后拼接的方式实现UNet3D网络"""
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.conv = CBR_Block_3x3(in_channels, mid_channels)
        self.bn = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU()

        self.encoder1 = EncoderBottleneck(mid_channels, mid_channels*2)
        self.encoder2 = EncoderBottleneck(mid_channels*2, mid_channels*4)
        self.encoder3 = EncoderBottleneck(mid_channels*4, mid_channels*8)
        self.encoder4 = EncoderBottleneck(mid_channels*8, mid_channels*16)
        
        self.bottom_layer = D_Inception_Block(mid_channels*16, mid_channels*8)

        self.decoder1 = DecoderBottleneck(mid_channels*16, mid_channels*4)
        self.decoder2 = DecoderBottleneck(mid_channels*8, mid_channels*2)
        self.decoder3 = DecoderBottleneck(mid_channels*4, mid_channels)
        self.FusionMagic = FusionMagic(mid_channels, mid_channels)

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = DecoderBottleneck(mid_channels*2, mid_channels)

        self.final_conv = nn.Conv3d(mid_channels, out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.relu(self.bn(self.conv(x))) # 32
        
        # 编码器部分
        x2 = self.encoder1(x1) # 64
        x3 = self.encoder2(x2) # 128
        x4 = self.encoder3(x3) # 256

        out = self.encoder4(x4) # 512
        
        # 特征增强
        out = self.bottom_layer(out)

        # 解码器部分
        out1 = self.decoder1(out, x4) # 128
        out2 = self.decoder2(out1, x3) # 64
        out3 = self.decoder3(out2, x2) # 32

        # # FusionMagic
        # FM_out = self.FusionMagic([out1, out2, out3])
        # y = FM_out.expand_as(out3)
        # out3 = y * out3  # 32
        # out3 = self.fusion_conv(out3)

        out4 = self.decoder4(out3, x1) # 16
        out = self.final_conv(out4)
        out = self.softmax(out)

        return out

class UNET3D_v3(nn.Module):
    """使用先上采样后拼接的方式实现UNet3D网络"""
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.conv = CBR_Block_3x3(in_channels, mid_channels)
        self.bn = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU()

        self.encoder1 = EncoderBottleneck(mid_channels, mid_channels*2)
        self.encoder2 = EncoderBottleneck(mid_channels*2, mid_channels*4)
        self.encoder3 = EncoderBottleneck(mid_channels*4, mid_channels*8)
        self.encoder4 = EncoderBottleneck(mid_channels*8, mid_channels*16)
        
        self.bottom_layer = D_Inception_Block(mid_channels*16, mid_channels*8)

        self.decoder1 = DecoderBottleneck(mid_channels*16, mid_channels*4, upsample=False)
        self.decoder2 = DecoderBottleneck(mid_channels*8, mid_channels*2, upsample=False)
        self.decoder3 = DecoderBottleneck(mid_channels*4, mid_channels, upsample=False)
        self.FusionMagic = FusionMagic_v2(mid_channels, mid_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = DecoderBottleneck(mid_channels*2, mid_channels, upsample=False)

        self.final_conv = nn.Conv3d(mid_channels, out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.relu(self.bn(self.conv(x))) # 32
        
        # 编码器部分
        x2 = self.encoder1(x1) # 64
        x3 = self.encoder2(x2) # 128
        x4 = self.encoder3(x3) # 256

        out = self.encoder4(x4) # 512
        
        # 特征增强
        out = self.bottom_layer(out)

        # 解码器部分
        out1 = self.decoder1(out, x4) # 128
        out2 = self.decoder2(out1, x3) # 64
        out3 = self.decoder3(out2, x2) # 32

        # # FusionMagic
        # FM_out = self.FusionMagic([out1, out2, out3])
        # y = FM_out.expand_as(out3)
        # out3 = y * out3  # 32
        # out3 = self.fusion_conv(out3)

        out4 = self.decoder4(out3, x1) # 16
        out = self.final_conv(out4)
        out = self.softmax(out)

        return out


   

if __name__ == '__main__':
    from Modules.UnetBlocks3D import *
    from Modules.Attentions3D import *
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RIA_UNET3D_v3(in_channels=4, out_channels=4, features=[16, 32, 64, 128, 256])
    # model = Down_SE2_UNET3D_v2(in_channels=4, mid_channels=32, out_channels=4)
    # model = Inception_Block(256, 128)
    # input = torch.rand((1, 256, 64, 64, 64)).to(device)
    input = torch.rand((1, 4, 128, 128, 128)).to(device)

    model = model.to(device)
    print(model)
    output = model(input)
    print(output.shape)
    summary(model, (1, 4, 128, 128, 128))
