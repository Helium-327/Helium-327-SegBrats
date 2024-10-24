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
# from Modules.UnetBlocks3D import *

# from Modules.Attentions3D import *

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
        
'''========================================== UNET3D网络 + CBAM ============================================'''
'''
 Finished:
    - ✅ F_CAC_UNET3D:  encoder和decoder均加入cbam   | DDL: 2024//
    - ✅ Down_CAC_UNET3D:  encoder加入cbam   | DDL: 2024//
    - ✅ Up_CAC_UNET3D:  decoder加入cbam   | DDL: 2024//
'''
class F_CAC_UNET3D(nn.Module):
    """
    UNET3D网络 + CBAM 结构:
    Full: encoder和decoder均加入cbam
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 features=[32, 64, 128, 256],
                #  down_att =False,
                #  up_att =False,
                #  bottom_att = False
                 ):
        super(F_CAC_UNET3D, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        # self.down_att    = down_att
        # self.up_att      = up_att
        # self.bottom_att  = bottom_att
        self.encoders_features  = features
        self.decoders_features  = (features + [features[-1]*2])[::-1]

        self.encoders     = nn.ModuleList()
        self.decoders     = nn.ModuleList()
        self._make_encoders()

        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(features[-1], features[-1]*2),
            CBAM(features[-1]*2),
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
                # 在两层卷积之间添加注意力机制， 聚焦重要的信息
                self.encoders.append(CBAM(self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            else:
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i-1], self.encoders_features[i]))
                # 在两层卷积之间添加注意力机制， 聚焦重要的信息
                self.encoders.append(CBAM(self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            # if self.down_att:
            #     self.encoders.append(CBAM(self.encoders_features[i]))
            self.encoders.append(DownSample())
            
    def _make_decoders(self):
        for i in range(len(self.decoders_features)):
            if i == len(self.decoders_features)-1:
                continue
            else:
                self.decoders.append(UpSample(self.decoders_features[i], self.decoders_features[i+1], trilinear=False))
                # CAC
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i], self.decoders_features[i+1]))
                self.decoders.append(CBAM(self.decoders_features[i+1]))
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i+1], self.decoders_features[i+1]))
            
    def forward(self, x):
        out = x
        # print(f'Input shape: {out.shape}')
        skip_out = []
        for m in self.encoders:
            if isinstance(m, DownSample):
                skip_out.append(out)
                # print(f"skip_out: {out.shape}")
            out = m(out)
            # print(f'Encoder shape: {out.shape}')
            # print("-" * 50)

        # for t in skip_out:
            # print(f'Skip connection shape: {t.shape}')

        out = self.bottom_layer(out)

        for m in self.decoders:
            if isinstance(m, UpSample):
                out = m(out)
                # print(f'up shape : {out.shape}')
                # print(f'skip shape : {skip_out[-1].shape}')
                out = torch.cat([out, skip_out.pop()], dim=1)
                # print(f'after cat shape : {out.shape}')
                # print("-" * 50)
            else:
                out = m(out)
                # print(f'Decoder shape: {out.shape}')
        out = self.out_conv(out)
        out = self.soft_max(out)
        # out = self.crf(out)
        return out
    
class Down_CAC_UNET3D(nn.Module):
    """
    UNET3D网络 + CBAM 结构:
    Down: encoder加入cbam
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 features=[32, 64, 128, 256],
                #  down_att =False,
                #  up_att =False,
                #  bottom_att = False
                 ):
        super(Down_CAC_UNET3D, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        # self.down_att    = down_att
        # self.up_att      = up_att
        # self.bottom_att  = bottom_att
        self.encoders_features  = features
        self.decoders_features  = (features + [features[-1]*2])[::-1]

        self.encoders     = nn.ModuleList()
        self.decoders     = nn.ModuleList()
        self._make_encoders()

        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(features[-1], features[-1]*2),
            CBAM(features[-1]*2),
            CBR_Block_3x3(features[-1]*2, features[-1]*2)
        )
        self._make_decoders()
        self.out_conv = nn.Conv3d(self.decoders_features[-1], self.out_channels, kernel_size=1)
        self.soft_max      = nn.Softmax(dim=1)

    def _make_encoders(self):
        for i in range(len(self.encoders_features)):
            if i == 0:
                self.encoders.append(CBR_Block_3x3(self.in_channels, self.encoders_features[i]))
                # 在两层卷积之间添加注意力机制， 聚焦重要的信息
                self.encoders.append(CBAM(self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            elif (i == len(self.encoders_features) -2) | (i == len(self.encoders_features) -1):
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i-1], self.encoders_features[i]))
                # 在两层卷积之间添加注意力机制， 聚焦重要的信息
                self.encoders.append(CBAM(self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            else:
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i-1], self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))

            # if self.down_att:
            #     self.encoders.append(CBAM(self.encoders_features[i]))
            self.encoders.append(DownSample())
            
    def _make_decoders(self):
        for i in range(len(self.decoders_features)):
            if i == len(self.decoders_features)-1:
                continue
            else:
                self.decoders.append(UpSample(self.decoders_features[i], self.decoders_features[i+1], trilinear=False))
                # CAC
                # self.decoders.append(CBAM(self.decoders_features[i+1]))
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i], self.decoders_features[i+1]))
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i+1], self.decoders_features[i+1]))
            
    def forward(self, x):
        out = x
        # print(f'Input shape: {out.shape}')
        skip_out = []
        for m in self.encoders:
            if isinstance(m, DownSample):
                skip_out.append(out)
                # print(f"skip_out: {out.shape}")
            out = m(out)
            # print(f'Encoder shape: {out.shape}')
            # print("-" * 50)

        # for t in skip_out:
            # print(f'Skip connection shape: {t.shape}')

        out = self.bottom_layer(out)

        for m in self.decoders:
            if isinstance(m, UpSample):
                out = m(out)
                # print(f'up shape : {out.shape}')
                # print(f'skip shape : {skip_out[-1].shape}')
                out = torch.cat([out, skip_out.pop()], dim=1)
                # print(f'after cat shape : {out.shape}')
                # print("-" * 50)
            else:
                out = m(out)
                # print(f'Decoder shape: {out.shape}')
        out = self.out_conv(out)
        return self.soft_max(out)
    
class Up_CAC_UNET3D(nn.Module):
    """
    UNET3D网络 + CBAM 结构:
    Down: decoder加入cbam
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 features=[32, 64, 128, 256],
                #  down_att =False,
                #  up_att =False,
                #  bottom_att = False
                 ):
        super(Up_CAC_UNET3D, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        # self.down_att    = down_att
        # self.up_att      = up_att
        # self.bottom_att  = bottom_att
        self.encoders_features  = features
        self.decoders_features  = (features + [features[-1]*2])[::-1]

        self.encoders     = nn.ModuleList()
        self.decoders     = nn.ModuleList()
        self._make_encoders()

        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(features[-1], features[-1]*2),
            CBAM(features[-1]*2),
            CBR_Block_3x3(features[-1]*2, features[-1]*2)
        )
        self._make_decoders()
        self.out_conv = nn.Conv3d(self.decoders_features[-1], self.out_channels, kernel_size=1)
        self.soft_max      = nn.Softmax(dim=1)

    def _make_encoders(self):
        for i in range(len(self.encoders_features)):
            if i == 0:
                self.encoders.append(CBR_Block_3x3(self.in_channels, self.encoders_features[i]))
                # 在两层卷积之间添加注意力机制， 聚焦重要的信息
                self.encoders.append(CBAM(self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            else:
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i-1], self.encoders_features[i]))
                # 在两层卷积之间添加注意力机制， 聚焦重要的信息
                self.encoders.append(CBAM(self.encoders_features[i]))
                self.encoders.append(CBR_Block_3x3(self.encoders_features[i], self.encoders_features[i]))
            # if self.down_att:
            #     self.encoders.append(CBAM(self.encoders_features[i]))
            self.encoders.append(DownSample())
            
    def _make_decoders(self):
        for i in range(len(self.decoders_features)):
            if i == len(self.decoders_features)-1:
                continue
            else:
                self.decoders.append(UpSample(self.decoders_features[i], self.decoders_features[i+1], trilinear=False))
                # CAC
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i], self.decoders_features[i+1]))
                # self.decoders.append(CBAM(self.decoders_features[i+1]))
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i+1], self.decoders_features[i+1]))
            
    def forward(self, x):
        out = x
        # print(f'Input shape: {out.shape}')
        skip_out = []
        for m in self.encoders:
            if isinstance(m, DownSample):
                skip_out.append(out)
                # print(f"skip_out: {out.shape}")
            out = m(out)
            # print(f'Encoder shape: {out.shape}')
            # print("-" * 50)

        # for t in skip_out:
            # print(f'Skip connection shape: {t.shape}')

        out = self.bottom_layer(out)

        for m in self.decoders:
            if isinstance(m, UpSample):
                out = m(out)
                # print(f'up shape : {out.shape}')
                # print(f'skip shape : {skip_out[-1].shape}')
                out = torch.cat([out, skip_out.pop()], dim=1)
                # print(f'after cat shape : {out.shape}')
                # print("-" * 50)
            else:
                out = m(out)
                # print(f'Decoder shape: {out.shape}')
        out = self.out_conv(out)
        return self.soft_max(out)

'''========================================== UNET3D网络 + Res ============================================'''
'''
TODO:
    - ✅ ResAttCBR_3x3 :包含Res结构的UNET3D网络    | DDL: 2024//
'''
        
class Res_UNET3D(nn.Module):
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
        super(Res_UNET3D, self).__init__()
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
            Inception_Block(features[-1], features[-1]*2),
            CBR_Block_3x3(features[-1]*2, features[-1]*2),
        )
        self._make_decoders()
        self.out_conv = nn.Conv3d(self.decoders_features[-1], self.out_channels, kernel_size=1)
        self.soft_max      = nn.Softmax(dim=1)

    def _make_encoders(self):
        for i in range(len(self.encoders_features)):
            if i == 0:
                self.encoders.append(ResAttCBR_3x3(self.in_channels, self.encoders_features[i]))
            else:
                self.encoders.append(ResAttCBR_3x3(self.encoders_features[i-1], self.encoders_features[i]))
            self.encoders.append(DownSample())
            
    def _make_decoders(self):
        for i in range(len(self.decoders_features)):
            if i == len(self.decoders_features)-1:
                continue
            else:
                self.decoders.append(UpSample(self.decoders_features[i], self.decoders_features[i+1], trilinear=False))
                self.decoders.append(ResAttCBR_3x3(self.decoders_features[i], self.decoders_features[i+1]))

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
        return self.soft_max(out)

'''========================================== UNET3D网络 + Inception ============================================'''
class RIA_UNET3D(nn.Module):
    """RID_UNET3D网络:
    RID: Residual Inception Attention
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
        super(RIA_UNET3D, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.down_att    = down_att
        self.up_att      = up_att
        self.bottom_att  = bottom_att
        self.encoders_features  = features
        self.decoders_features  = (features + [features[-1]*2])[::-1]

        self.encoders     = nn.ModuleList()
        self.decoders     = nn.ModuleList()
        self.skippers = nn.ModuleList()

        self._make_encoders()
        self._make_skip_conv_layers()
        self.bottom_layer = nn.Sequential(
            Inception_Block(features[-1], features[-1]*2),
        )
        self._make_decoders()
        self.out_conv = nn.Conv3d(self.decoders_features[-1], self.out_channels, kernel_size=1)
        self.soft_max = nn.Softmax(dim=1)
        # self.crf      = CRF(out_channels)

    def _make_encoders(self):
        for i in range(len(self.encoders_features)):
            if i == 0:
                self.encoders.append(Inception_Block(self.out_channels, self.encoders_features[i]))
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
                self.decoders.append(CBAM(self.decoders_features[i+1]))
                self.decoders.append(CBR_Block_3x3(self.decoders_features[i+1], self.decoders_features[i+1]))
    
    def _make_skip_conv_layers(self):
        for i in range(len(self.encoders_features)):
            if i == 0:
                self.skippers.append(nn.Sequential(
                    SelfAttention3D(self.encoders_features[i]),
                    nn.BatchNorm3d(self.encoders_features[i]),
                    nn.GELU(),
                    ))
            else:
                self.skippers.append(nn.Sequential(
                    SelfAttention3D(self.encoders_features[i]),
                    nn.GELU(),
                    nn.Conv3d(self.encoders_features[i], self.encoders_features[i-1], kernel_size=3, padding=1),
                    nn.BatchNorm3d(self.encoders_features[i-1]),
                    nn.ReLU(),
                    SelfAttention3D(self.encoders_features[i-1]),
                    nn.GELU(),
                    nn.Conv3d(self.encoders_features[i-1], self.encoders_features[i], kernel_size=1),
                    nn.BatchNorm3d(self.encoders_features[i]),
                ))

        
    def forward(self, x):
        out = x
        # print(f'Input shape: {out.shape}')
        skip_out = []
        
        for m in self.encoders:
            if isinstance(m, DownSample):
                skip_out.append(out)
                # print(f"skip_out: {out.shape}")
            out = m(out)
            # print(f'Encoder shape: {out.shape}')
            # print("-" * 50)

        for i, m in enumerate(self.skippers):
            # s_out = m(skip_out[i]) 
            s_out = m(skip_out[i]) + skip_out[i]
            skip_out[i] = F.gelu(s_out)

        # for t in skip_out:
        #     print(f'Skip connection shape: {t.shape}')

        
        out = self.bottom_layer(out)

        for m in self.decoders:
            if isinstance(m, UpSample):
                out = m(out)
                # print(f'up shape : {out.shape}')
                # print(f'skip shape : {skip_out[-1].shape}')

                out = torch.cat([out, skip_out.pop()], dim=1)
                # print(f'after cat shape : {out.shape}')
                # print("-" * 50)
            else:
                out = m(out)
                # print(f'Decoder shape: {out.shape}')
        out = self.out_conv(out)
        out = self.soft_max(out)
        # out = self.crf
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RIA_UNET3D(in_channels=4, out_channels=4, features=[16, 32, 64, 128, 256])
    # model = Inception_Block(256, 128)
    # input = torch.rand((1, 256, 64, 64, 64)).to(device)
    input = torch.rand((1, 4, 128, 128, 128)).to(device)

    model = model.to(device)
    print(model)
    output = model(input)
    print(output.shape)
    summary(model, (1, 4, 128, 128, 128))
