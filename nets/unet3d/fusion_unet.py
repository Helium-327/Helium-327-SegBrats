import skimage
import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn import functional as F

# from Modules.Attentions3D import *
# from Modules.UnetBlocks3D import *

from nets.unet3d.Modules.UnetBlocks3D import *
from nets.unet3d.Modules.Attentions3D import *
from nets.unet3d.unet3d import DoubleCBR_Block_3x3

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=False):
        super().__init__()

        if dilation:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.residual = nn.Sequential( 
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
            )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual_out = self.residual(x)
        shortcut_out = self.shortcut(x)
        out = self.relu(residual_out + shortcut_out)
        return out

class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=False):
        super().__init__()

        if dilation:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
            )
        
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.cbam = CBAM(in_channels//2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skipped=None):
        if skipped is not None:
            # skipped = self.cbam(skipped)
            x = torch.cat([x, skipped], dim=1)

        residual_out = self.residual(x)
        shortcut_out = self.shortcut(x)
        out = self.relu(residual_out + shortcut_out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=4, dilation_flags=[False, False, False, False]):
        super().__init__()
        self.DownSample = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder1 = EncoderBottleneck(in_channels, 32, dilation_flags[0])     
        self.encoder2 = EncoderBottleneck(32, 64, dilation_flags[1])
        self.encoder3 = EncoderBottleneck(64, 128, dilation_flags[2])
        self.encoder4 = EncoderBottleneck(128, 256, dilation_flags[3])
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out1 = self.encoder1(x)
        out = self.DownSample(out1)

        out2 = self.encoder2(out)
        out = self.DownSample(out2)

        out3 = self.encoder3(out)
        out = self.DownSample(out3)

        out4 = self.encoder4(out)
        out = self.DownSample(out4)

        out = self.dropout(out)
        skip_connections = [out1, out2, out3, out4] # 32x128 64x64 128x32 256x16

        return out, skip_connections

class Decoder(nn.Module):
    def __init__(self, out_channels, dilation_flags=[False, False, False, False], fusion=False):
        super().__init__()
        self.fusion = fusion

        self.upsample1 = nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1)
        self.decoder1 = DecoderBottleneck(512, 128, dilation_flags[0])

        self.upsample2 = nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1)
        self.decoder2 = DecoderBottleneck(256, 64, dilation_flags[1])

        self.upsample3 = nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1)
        self.decoder3 = DecoderBottleneck(128, 32, dilation_flags[2])

        self.upsample4 = nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1)
        self.decoder4 = DecoderBottleneck(64, 32, dilation_flags[3])

        self.FusionMagic = FusionMagic_v2(32, 32)

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        
        skip_connections = skip_connections[::-1]

        out = self.upsample1(x) # 256x8 --> 256 x16
        out1 = self.decoder1(out, skip_connections[0]) # (256 + 256)x16 --> 128x32

        out = self.upsample2(out1) 
        out2 = self.decoder2(out, skip_connections[1]) # (128 + 128)x32 --> 64x64

        out = self.upsample3(out2)
        out3 = self.decoder3(out, skip_connections[2]) # (64 + 64)x64 --> 32x128

        if self.fusion:
            out3 = self.FusionMagic([out1, out2, out3])

        out = self.upsample4(out3)
        out4 = self.decoder4(out, skip_connections[3]) # (32 + 32)x128 --> 32x256
        
        out = self.out_conv(out4)

        return out


class FM_UNET3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, fusion=False, ec_dilation_flags=[False, False, False, False], dc_dilation_flags=[False, False, False, False]):
        super(FM_UNET3D, self).__init__()
        self.encoders = Encoder(in_channels, ec_dilation_flags)

        self.upsample = nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1)

        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(256, 512),
            CBAM(512, 256),
            CBR_Block_3x3(512, 256)
        )

        self.decoders = Decoder(out_channels, dc_dilation_flags, fusion=fusion)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        enc_out, skip_connections = self.encoders(x)
        bottom_out = self.bottom_layer(enc_out)
        dec_out = self.decoders(bottom_out, skip_connections)

        out = self.softmax(dec_out)
        return out

class FM_UNET3_v1(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(FM_UNET3D, self).__init__()
        self.encoders = Encoder(in_channels)

        self.upsample = nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bottom_layer = nn.Sequential(
            DoubleCBR_Block_3x3(256, 512),
            SE_Block(512),
            DoubleCBR_Block_3x3(512, 256)
        )

        self.decoders = Decoder(out_channels)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        enc_out, skip_connections = self.encoders(x)
        bottom_out = self.bottom_layer(enc_out)
        dec_out = self.decoders(bottom_out, skip_connections)

        out = self.softmax(dec_out)
        return out
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = FM_UNET3D(in_channels=4, out_channels=4).to(device)
    input_1 = torch.rand((1, 4, 128, 128, 128)).to(device)

    out = model(input_1)

    summary(model, (1, 4, 128, 128, 128))
