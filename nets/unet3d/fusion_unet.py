import skimage
import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F

# from Modules.Attentions3D import *
# from Modules.UnetBlocks3D import *
from nets.unet3d.Modules.UnetBlocks3D import *
from nets.unet3d.Modules.Attentions3D import *

class FM_UNET3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, fusion=False, ec_dilation_flags=[False, False, False, False], dc_dilation_flags=[False, False, False, False]):
        super(FM_UNET3D, self).__init__()
        # 编码器
        self.fusion = fusion
        self.DownSample = nn.MaxPool3d(kernel_size=4, stride=2, padding=1)

        # 编码器（不适用cbam效果更好）
        self.encoder1 = EncoderBottleneck(in_channels, 32, 3, 1, 1, ec_dilation_flags[0])
        self.encoder2 = EncoderBottleneck(32, 64, 3, 1, 1, ec_dilation_flags[1])
        self.encoder3 = EncoderBottleneck(64, 128, 3, 1, 1, ec_dilation_flags[2])
        self.encoder4 = EncoderBottleneck(128, 256, 3, 1, 1, ec_dilation_flags[3])

        # 最低层
        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(256, 512),
            # CBAM(512, 256, 3), # 实验结果表明不使用CBAM效果更好
            CBR_Block_3x3(512, 256)
        )

        # 解码器
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1),
            # CBR_Block_3x3(256, 256)
        )
        self.decoder1 = DecoderBottleneck(512, 128, 3, 1, 1, dc_dilation_flags[0])

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),
            # CBR_Block_3x3(128, 128)
            )
        self.decoder2 = DecoderBottleneck(256, 64, 3, 1, 1, dc_dilation_flags[1])

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1),
            # CBR_Block_5x5(64, 64)
            )
        self.decoder3 = DecoderBottleneck(128, 32, 3, 1, 1, dc_dilation_flags[2])

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1),
            # CBR_Block_5x5(32, 32)
            )

        self.decoder4 = DecoderBottleneck(64, 32, 3, 1, 1, dc_dilation_flags[3]) 
        
        # 融合层
        self.FusionMagic = FusionMagic_v2(32, 32)

        self.dropout = nn.Dropout(p=0.2)

        self.out_conv = nn.Sequential(
            CBR_Block_3x3(32, 32),
            nn.Conv3d(32, out_channels, kernel_size=1)
            )

        self.encoders = Encoder(in_channels, ec_dilation_flags)

        self.upsample = nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ec_out1 = self.encoder1(x)
        out = self.DownSample(ec_out1)

        ec_out2 = self.encoder2(out)
        out = self.DownSample(ec_out2)

        ec_out3 = self.encoder3(out)
        out = self.DownSample(ec_out3)

        ec_out4 = self.encoder4(out)
        out = self.DownSample(ec_out4)

        skip_connections = [ec_out1, ec_out2, ec_out3, ec_out4] # 32x128 64x64 128x32 256x16

        bottom_out = self.bottom_layer(out)

        # 解码器
        skip_connections = skip_connections[::-1]

        out = self.upsample1(bottom_out) # 256x8 --> 256 x16
        out1 = self.decoder1(out, skip_connections[0]) # (256 + 256)x16 --> 128x32

        out = self.upsample2(out1) 
        out2 = self.decoder2(out, skip_connections[1]) # (128 + 128)x32 --> 64x64

        out = self.upsample3(out2)
        out3 = self.decoder3(out, skip_connections[2]) # (64 + 64)x64 --> 32x128

        # 融合层
        if self.fusion:
            out3 = self.FusionMagic([out1, out2, out3])
            out3 = self.dropout(out3)

        out = self.upsample4(out3)
        out4 = self.decoder4(out, skip_connections[3]) # (32 + 32)x128 --> 32x256
        
        out = self.out_conv(out4)

        out = self.softmax(out)
        return out


class FM_UNET3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, fusion=False, ec_dilation_flags=[False, False, False, False], dc_dilation_flags=[False, False, False, False], residuals_flag=False):
        super(FM_UNET3D, self).__init__()
        # 编码器
        self.fusion = fusion
        self.DownSample = nn.MaxPool3d(kernel_size=4, stride=2, padding=1)

        # 编码器（不适用cbam效果更好）
        self.encoder1 = EncoderBottleneck(in_channels, 32, 3, 1, 1, ec_dilation_flags[0], residual=residuals_flag)
        self.encoder2 = EncoderBottleneck(32, 64, 3, 1, 1, ec_dilation_flags[1], residual=residuals_flag)
        self.encoder3 = EncoderBottleneck(64, 128, 3, 1, 1, ec_dilation_flags[2], residual=residuals_flag)
        self.encoder4 = EncoderBottleneck(128, 256, 3, 1, 1, ec_dilation_flags[3], residual=residuals_flag)

        # 最低层
        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(256, 512),
            # CBAM(512, 256, 3), # 实验结果表明不使用CBAM效果更好
            CBR_Block_3x3(512, 256)
        )

        # 解码器
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1),
            # CBR_Block_3x3(256, 256)
        )
        self.decoder1 = DecoderBottleneck(512, 128, 3, 1, 1, dc_dilation_flags[0], residual=residuals_flag)

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),
            # CBR_Block_3x3(128, 128)
            )
        self.decoder2 = DecoderBottleneck(256, 64, 3, 1, 1, dc_dilation_flags[1], residual=residuals_flag)

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1),
            # CBR_Block_5x5(64, 64)
            )
        self.decoder3 = DecoderBottleneck(128, 32, 3, 1, 1, dc_dilation_flags[2], residual=residuals_flag)

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1),
            # CBR_Block_5x5(32, 32)
            )

        self.decoder4 = DecoderBottleneck(64, 32, 3, 1, 1, dc_dilation_flags[3], residual=residuals_flag) 
        
        # 融合层
        self.FusionMagic = FusionMagic_v2(32, 32)

        self.dropout = nn.Dropout(p=0.2)

        self.out_conv = nn.Sequential(
            CBR_Block_3x3(32, 32),
            nn.Conv3d(32, out_channels, kernel_size=1)
            )

        self.encoders = Encoder(in_channels, ec_dilation_flags)

        self.upsample = nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ec_out1 = self.encoder1(x)
        out = self.DownSample(ec_out1)

        ec_out2 = self.encoder2(out)
        out = self.DownSample(ec_out2)

        ec_out3 = self.encoder3(out)
        out = self.DownSample(ec_out3)

        ec_out4 = self.encoder4(out)
        out = self.DownSample(ec_out4)

        skip_connections = [ec_out1, ec_out2, ec_out3, ec_out4] # 32x128 64x64 128x32 256x16

        bottom_out = self.bottom_layer(out)

        # 解码器
        skip_connections = skip_connections[::-1]

        out = self.upsample1(bottom_out) # 256x8 --> 256 x16
        out1 = self.decoder1(out, skip_connections[0]) # (256 + 256)x16 --> 128x32

        out = self.upsample2(out1) 
        out2 = self.decoder2(out, skip_connections[1]) # (128 + 128)x32 --> 64x64

        out = self.upsample3(out2)
        out3 = self.decoder3(out, skip_connections[2]) # (64 + 64)x64 --> 32x128

        # 融合层
        if self.fusion:
            out3 = self.FusionMagic([out1, out2, out3])
            out3 = self.dropout(out3)

        out = self.upsample4(out3)
        out4 = self.decoder4(out, skip_connections[3]) # (32 + 32)x128 --> 32x256
        
        out = self.out_conv(out4)

        out = self.softmax(out)
        return out

class FM_UNET3D_v1(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, fusion=False, ec_dilation_flags=[False, False, False, False], dc_dilation_flags=[False, False, False, False]):
        super(FM_UNET3D_v1, self).__init__()
        # 编码器
        self.fusion = fusion
        self.DownSample = nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
        self.encoder1 = EncoderBottleneck(in_channels, 32, 3, 1, 1, ec_dilation_flags[0])      # 增大感受野#? 增大感受野会导致模型收敛变慢
        self.encoder2 = EncoderBottleneck(32, 64, 3, 2, 1, ec_dilation_flags[1])
        self.encoder3 = EncoderBottleneck(64, 128, 3, 1, 1, ec_dilation_flags[2])
        self.encoder4 = EncoderBottleneck(128, 256, 3, 1, 1, ec_dilation_flags[3])
        self.dropout = nn.Dropout(p=0.2)

        # 最低层
        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(256, 512),
            nn.Dropout(p=0.2),
            CBAM(512, 256),
            CBR_Block_3x3(512, 256)
        )

        # 解码器
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1),
            CBR_Block_3x3(256, 256)
        )
        self.decoder1 = DecoderBottleneck(512, 128, 3, 1, 1, dc_dilation_flags[0])

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),
            CBR_Block_3x3(128, 128)
            )
        self.decoder2 = DecoderBottleneck(256, 64, 3, 1, 1, dc_dilation_flags[1])

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1),
            CBR_Block_5x5(64, 64)
            )
        self.decoder3 = DecoderBottleneck(128, 32, 3, 1, 1, dc_dilation_flags[2])

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1),
            CBR_Block_5x5(32, 32)
            )

        self.decoder4 = DecoderBottleneck(64, 32, 3, 1, 1, dc_dilation_flags[3]) # 增大感受野
        
        # 融合层
        self.FusionMagic = FusionMagic_v2(32, 32)

        self.dropout = nn.Dropout(p=0.2)

        self.out_conv = nn.Sequential(
            CBR_Block_3x3(32, 32),
            nn.Conv3d(32, out_channels, kernel_size=1)
            )

        self.encoders = Encoder(in_channels, ec_dilation_flags)

        self.upsample = nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        ec_out1 = self.encoder1(x)
        out = self.DownSample(ec_out1)

        ec_out2 = self.encoder2(out)
        out = self.DownSample(ec_out2)

        ec_out3 = self.encoder3(out)
        out = self.DownSample(ec_out3)

        ec_out4 = self.encoder4(out)
        out = self.DownSample(ec_out4)
        
        out = self.dropout(out)

        skip_connections = [ec_out1, ec_out2, ec_out3, ec_out4] # 32x128 64x64 128x32 256x16

        bottom_out = self.bottom_layer(out)

        # 解码器
        skip_connections = skip_connections[::-1]

        out = self.upsample1(bottom_out) # 256x8 --> 256 x16
        out1 = self.decoder1(out, skip_connections[0]) # (256 + 256)x16 --> 128x32

        out = self.upsample2(out1) 
        out2 = self.decoder2(out, skip_connections[1]) # (128 + 128)x32 --> 64x64

        out = self.upsample3(out2)
        out3 = self.decoder3(out, skip_connections[2]) # (64 + 64)x64 --> 32x128

        # 融合层
        if self.fusion:
            out3 = self.FusionMagic([out1, out2, out3])

        out = self.upsample4(out3)
        out4 = self.decoder4(out, skip_connections[3]) # (32 + 32)x128 --> 32x256
        
        out = self.out_conv(out4)


        out = self.softmax(out)
        return out
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = FM_UNET3D(in_channels=4, out_channels=4).to(device)
    input_1 = torch.rand((1, 4, 128, 128, 128)).to(device)

    out = model(input_1)

    summary(model, (1, 4, 128, 128, 128))
