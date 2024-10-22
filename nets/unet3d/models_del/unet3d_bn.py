# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/08 19:02:54
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 使用bn的3D UNet
=================================================

================================================================
Total params: 47,383,620
Trainable params: 47,383,620
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 32.00
Forward/backward pass size (MB): 16624.00
Params size (MB): 180.75
Estimated Total Size (MB): 16836.75
----------------------------------------------------------------
'''

import torch
from torchinfo import summary
import torch.nn as nn
from torch.nn import functional as F
from nets.unet3d.ref.modules import Up_Block
from nets.unet3d.attentions.SE import SE_Block
from nets.unet3d.ref.CBR_Blocks import *

# from torchsummary import summary

class UNet3D_BN(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 dropout_rate:float=0.2, 
                 use_dropout:bool=True, 
                 ln_spatial_shape:list=[],
                 features = [32, 64, 128, 256]):
        super(UNet3D_BN, self).__init__()
        self.encoder_features = features
        self.decoder_features = (features + [features[-1]*2])[::-1]
        self.up_features = self.decoder_features
        self.bottom_layer = nn.Sequential(
            CBR_Block_3x3(self.encoder_features[-1], self.encoder_features[-1]*2),
            CBR_Block_3x3(self.encoder_features[-1]*2, self.encoder_features[-1]*2)
        )
        # 编码器
        self.encoders = nn.ModuleDict()
        self.up_layers = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        # 构建编码器
        for i in range(len(self.encoder_features)):
            if i == 0:
                self.encoders[f'encoder{i}_to_{self.encoder_features[i]}'] = nn.Sequential(
                    CBR_Block_3x3(in_channels, self.encoder_features[i]),
                    CBR_Block_3x3(self.encoder_features[i], self.encoder_features[i])
                    )
            else:
                self.encoders[f'encoder{i}_to_{self.encoder_features[i]}'] = nn.Sequential(
                    CBR_Block_3x3(self.encoder_features[i-1], self.encoder_features[i]),
                    CBR_Block_3x3(self.encoder_features[i],self. encoder_features[i])
                    )

        # 构建解码器
        for i in range(len(self.decoder_features)):
            if i == len(self.decoder_features) -1:
                self.decoders[f'decoder{i}_to_{out_channels}'] = nn.Conv3d(self.decoder_features[i], out_channels, kernel_size=1)
            else:
                self.decoders[f'decoder{i}_to_{self.decoder_features[i+1]}'] = nn.Sequential(
                    CBR_Block_3x3(self.decoder_features[i], self.decoder_features[i+1]),
                    CBR_Block_3x3(self.decoder_features[i+1], self.decoder_features[i+1])
                    )
    
        for i in range(len(self.up_features)-1):
            self.up_layers[f'up{i}_to_{self.up_features[i+1]}'] = nn.Sequential(
                Up_Block(self.up_features[i], self.up_features[i], 4, 2, 1),
                CBR_Block_3x3(self.up_features[i], self.up_features[i+1]))
        # 输出层
        self.output_conv = nn.Conv3d(self.encoder_features[0], out_channels, kernel_size=1)

        # 归一化层
        self.dropout = nn.Dropout3d(dropout_rate)

        self.soft = nn.Softmax(dim=1)
        # print(self.encoders)
        # print(self.up_layers)
        # print(self.decoders)
    
    def forward(self, x):
        skip_out_list = []
        # 编码器
        for i, (module_name, module) in enumerate(self.encoders.items()):
            if i == 0:                
                skip_out = module(x) 
                # print(f"encoder module name: {module_name}")
                # print(skip_out.shape)
            else:
                skip_out = module(F.max_pool3d(skip_out, 2, 2))
                # print(f"encoder module name: {module_name}")
                # print(skip_out.shape)
            skip_out_list.append(skip_out)
            out = skip_out
        
        # bottom layers
        out = self.bottom_layer(out)
        out = F.max_pool3d(out, 2, 2)

        # 解码器
        for i, ((d_module_name, d_module), (up_module_name, up_module)) in enumerate(zip(self.decoders.items(), self.up_layers.items())):
            if i < len(self.decoder_features):
                out = d_module(torch.cat([up_module(out), skip_out_list.pop()], dim=1))
                # print(d_module_name)
                # print(up_module_name)
                # print(out.shape)
        out = self.output_conv(out)
        out = self.soft(out)
        return out


# class UNet3D_BN(nn.Module):
#     def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0.2, use_dropout:bool=True, ln_spatial_shape:list=[]):
#         super(UNet3D_BN, self).__init__()
#         self.dropout_rate = dropout_rate
#         self.use_dropout = use_dropout
#         # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
#         # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
#         # 编码器
#         self.encoder1 = nn.Sequential(
#             CBR_Block_3x3(in_channels, 32),
#             CBR_Block_3x3(32, 32),
#         )
#         self.encoder2 = nn.Sequential(
#             CBR_Block_3x3(32, 64),
#             CBR_Block_3x3(64, 64),
#         )
#         self.encoder3 = nn.Sequential(
#             CBR_Block_3x3(64, 128),
#             CBR_Block_3x3(128, 128),
#         )
#         self.encoder4 = nn.Sequential(
#             CBR_Block_3x3(128, 256),
#             CBR_Block_3x3(256, 256),
#         )
#         self.encoder5 = nn.Sequential(
#             CBR_Block_3x3(256, 512),
#             CBR_Block_3x3(512, 512),
#         )
#         # 解码器
#         self.decoder1 = nn.Sequential(
#             CBR_Block_3x3(512, 256),
#             CBR_Block_3x3(256, 256),
#         )
#         self.up1      = nn.Sequential(
#             Up_Block(512, 512, 4, 2, 1),
#             CBR_Block_3x3(512, 256),
#         )

#         self.decoder2 = nn.Sequential(
#             CBR_Block_3x3(256, 128),
#             CBR_Block_3x3(128, 128),
#         )
#         self.up2      = nn.Sequential(
#             Up_Block(256, 256, 4, 2, 1),
#             CBR_Block_3x3(256, 128),
#         )

#         self.decoder3 = nn.Sequential(
#             CBR_Block_3x3(128, 64),
#             CBR_Block_3x3(64, 64),
#         )
#         self.up3      = nn.Sequential(
#             Up_Block(128, 64, 4, 2, 1),
#             CBR_Block_3x3(64, 64),
#         )

#         self.decoder4 = nn.Sequential(
#             CBR_Block_3x3(64, 32),
#             CBR_Block_3x3(32, 32),
#         )
#         self.up4      = nn.Sequential(
#             Up_Block(64, 32, 4, 2, 1),
#             CBR_Block_3x3(32, 32),
#         )

#         # 输出层
#         self.output_conv = nn.Conv3d(32, out_channels, kernel_size=1)

#         # 归一化层
#         self.dropout = nn.Dropout3d(dropout_rate)

#         self.soft = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         # 编码器
#         t1 = self.encoder1(x)                                                                   # [1, 32, 128, 128, 128]
#         t2 = self.encoder2(F.max_pool3d(t1, 2, 2))                                              # [1, 64, 64, 64, 64] 
#         t3 = self.encoder3(F.max_pool3d(t2, 2, 2))                                              # [1, 128, 32, 32, 32]
#         t4 = self.encoder4(F.max_pool3d(t3, 2, 2))                                              # [1, 256, 16, 16, 16]
#         out = self.encoder5(F.max_pool3d(t4, 2, 2))                                             # [1, 512, 8, 8, 8]

#         # out = self.dropout(out) if self.use_dropout else out
#         # 解码器        
#         out = self.decoder1(torch.cat([self.up1(out), t4], dim=1))                               # [1, 256, 16, 16, 16]
#         out = self.decoder2(torch.cat([self.up2(out), t3], dim=1))                               # [1, 128, 32, 32, 32]                      
#         out = self.decoder3(torch.cat([self.up3(out), t2], dim=1))                               # [1, 64, 64, 64, 64]
#         out = self.decoder4(torch.cat([self.up4(out), t1], dim=1))                               # [1, 32, 128, 128, 128]

#         # 输出层
#         out = self.output_conv(out)
#         out = self.soft(out)

#         return out
    
class UNet3D_BN_SE(nn.Module):
    """
    使用SE注意力机制的3D UNet
    ================================================================
    Total params: 47,412,964
    Trainable params: 47,412,964
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 32.00
    Forward/backward pass size (MB): 16626.01
    Params size (MB): 180.87
    Estimated Total Size (MB): 16838.88
    ----------------------------------------------------------------
    """
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0.2, use_dropout:bool=True, ln_spatial_shape:list=[]):
        super(UNet3D_BN_SE, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
        # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
        # 编码器
        self.encoder1 = nn.Sequential(
            CBR_Block_3x3(in_channels, 32),
            CBR_Block_3x3(32, 32),
        )
        self.encoder2 = nn.Sequential(
            CBR_Block_3x3(32, 64),
            CBR_Block_3x3(64, 64),
        )
        self.encoder3 = nn.Sequential(
            CBR_Block_3x3(64, 128),
            CBR_Block_3x3(128, 128),
        )
        
        self.encoder4 = nn.Sequential(
            CBR_Block_3x3(128, 256),
            CBR_Block_3x3(256, 256),
        )

        self.encoder5 = nn.Sequential(
            CBR_Block_3x3(256, 512),
            CBR_Block_3x3(512, 512),
        )

        self.se_attention = SE_Block(512, 16)

        # 解码器
        self.decoder1 = nn.Sequential(
            CBR_Block_3x3(512, 256),
            CBR_Block_3x3(256, 256),
        )
        self.up1      = nn.Sequential(
            Up_Block(512, 512, 4, 2, 1),
            CBR_Block_3x3(512, 256),
        )


        self.decoder2 = nn.Sequential(
            CBR_Block_3x3(256, 128),
            CBR_Block_3x3(128, 128),
        )
        self.up2      = nn.Sequential(
            Up_Block(256, 256, 4, 2, 1),
            CBR_Block_3x3(256, 128),
        )


        self.decoder3 = nn.Sequential(
            CBR_Block_3x3(128, 64),
            CBR_Block_3x3(64, 64),
        )
        self.up3      = nn.Sequential(
            Up_Block(128, 64, 4, 2, 1),
            CBR_Block_3x3(64, 64),
        )

        self.decoder4 = nn.Sequential(
            CBR_Block_3x3(64, 32),
            CBR_Block_3x3(32, 32),
        )
        self.up4      = nn.Sequential(
            Up_Block(64, 32, 4, 2, 1),
            CBR_Block_3x3(32, 32),
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

        out = self.se_attention(out)

        # out = self.dropout(out) if self.use_dropout else out
        # 解码器        
        out = self.decoder1(torch.cat([self.up1(out), t4], dim=1))                               # [1, 256, 16, 16, 16]
        out = self.decoder2(torch.cat([self.up2(out), t3], dim=1))                               # [1, 128, 32, 32, 32]                      
        out = self.decoder3(torch.cat([self.up3(out), t2], dim=1))                               # [1, 64, 64, 64, 64]
        out = self.decoder4(torch.cat([self.up4(out), t1], dim=1))                               # [1, 32, 128, 128, 128]

        # 输出层
        out = self.output_conv(out)
        out = self.soft(out)

        return out




class UNet3D_ResBN(nn.Module):
    """
    使用BN和Res_block的3D UNet
    参数
    ================================================================
    Total params: 56,956,516
    Trainable params: 56,956,516
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 32.00
    Forward/backward pass size (MB): 17474.00
    Params size (MB): 217.27
    Estimated Total Size (MB): 17723.27
    ----------------------------------------------------------------
    """
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0, use_bn:bool=True, use_ln:bool=False, use_dropout:bool=False, ln_spatial_shape:list=[]):
        super(UNet3D_ResBN, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
        # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
        # 编码器
        self.encoder1 = nn.Sequential(
            CBR_Block_3x3(in_channels, 32),
            CBR_Block_3x3(32, 32),
        )
        self.encoder2 = nn.Sequential(
            ResCBR_3x3(32, 64),
            # CBR_Block_3x3(64, 64),
        )
        self.encoder3 = nn.Sequential(
            ResCBR_3x3(64, 128),
            # CBR_Block_3x3(128, 128),
        )
        self.encoder4 = nn.Sequential(
            ResCBR_3x3(128, 256),
            # CBR_Block_3x3(256, 256),
        )
        self.encoder5 = nn.Sequential(
            ResCBR_3x3(256, 512),
            # CBR_Block_3x3(512, 512),
        )

        # 解码器
        self.decoder1 = nn.Sequential(
            CBR_Block_3x3(512, 256),
            CBR_Block_3x3(256, 256),
        )
        self.up1      = nn.Sequential(
            Up_Block(512, 512, 4, 2, 1),
            CBR_Block_3x3(512, 256),
        )


        self.decoder2 = nn.Sequential(
            CBR_Block_3x3(256, 128),
            CBR_Block_3x3(128, 128),
        )
        self.up2      = nn.Sequential(
            Up_Block(256, 256, 4, 2, 1),
            CBR_Block_3x3(256, 128),
        )


        self.decoder3 = nn.Sequential(
            CBR_Block_3x3(128, 64),
            CBR_Block_3x3(64, 64),
        )
        self.up3      = nn.Sequential(
            Up_Block(128, 64, 4, 2, 1),
            CBR_Block_3x3(64, 64),
        )

        self.decoder4 = nn.Sequential(
            CBR_Block_3x3(64, 32),
            CBR_Block_3x3(32, 32),
        )
        self.up4      = nn.Sequential(
            Up_Block(64, 32, 4, 2, 1),
            CBR_Block_3x3(32, 32),
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

        # 解码器        
        out = self.decoder1(torch.cat([self.up1(out), t4], dim=1))                               # [1, 256, 16, 16, 16]
        out = self.decoder2(torch.cat([self.up2(out), t3], dim=1))                               # [1, 128, 32, 32, 32]                      
        out = self.decoder3(torch.cat([self.up3(out), t2], dim=1))                               # [1, 64, 64, 64, 64]
        out = self.decoder4(torch.cat([self.up4(out), t1], dim=1))                               # [1, 32, 128, 128, 128]

        # 输出层
        out = self.output_conv(out)
        out = self.soft(out)

        return out

class UNet3D_ResBN_SE(nn.Module):
    """
    参数
    ================================================================
    Total params: 56,989,284
    Trainable params: 56,989,284
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 32.00
    Forward/backward pass size (MB): 17476.01
    Params size (MB): 217.40
    Estimated Total Size (MB): 17725.41
    ----------------------------------------------------------------
    """
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0, use_bn:bool=True, use_ln:bool=False, use_dropout:bool=False, ln_spatial_shape:list=[]):
        super(UNet3D_ResBN_SE, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
        # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
        # 编码器
        self.encoder1 = nn.Sequential(
            CBR_Block_3x3(in_channels, 32),
            CBR_Block_3x3(32, 32),
        )
        self.encoder2 = nn.Sequential(
            ResCBR_3x3(32, 64),
            # CBR_Block_3x3(64, 64),
        )
        self.encoder3 = nn.Sequential(
            ResCBR_3x3(64, 128),
            # CBR_Block_3x3(128, 128),
        )
        self.encoder4 = nn.Sequential(
            ResCBR_3x3(128, 256),
            # CBR_Block_3x3(256, 256),
        )
        self.encoder5 = nn.Sequential(
            ResCBR_3x3(256, 512),
            # CBR_Block_3x3(512, 512),
        )

        self.se_attention = SE_Block(512, 16)
        # 解码器
        self.decoder1 = nn.Sequential(
            CBR_Block_3x3(512, 256),
            CBR_Block_3x3(256, 256),
        )
        self.up1      = nn.Sequential(
            Up_Block(512, 512, 4, 2, 1),
            CBR_Block_3x3(512, 256),
        )


        self.decoder2 = nn.Sequential(
            CBR_Block_3x3(256, 128),
            CBR_Block_3x3(128, 128),
        )
        self.up2      = nn.Sequential(
            Up_Block(256, 256, 4, 2, 1),
            CBR_Block_3x3(256, 128),
        )


        self.decoder3 = nn.Sequential(
            CBR_Block_3x3(128, 64),
            CBR_Block_3x3(64, 64),
        )
        self.up3      = nn.Sequential(
            Up_Block(128, 64, 4, 2, 1),
            CBR_Block_3x3(64, 64),
        )

        self.decoder4 = nn.Sequential(
            CBR_Block_3x3(64, 32),
            CBR_Block_3x3(32, 32),
        )
        self.up4      = nn.Sequential(
            Up_Block(64, 32, 4, 2, 1),
            CBR_Block_3x3(32, 32),
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

        out = self.se_attention(out)

        # 解码器        
        out = self.decoder1(torch.cat([self.up1(out), t4], dim=1))                               # [1, 256, 16, 16, 16]
        out = self.decoder2(torch.cat([self.up2(out), t3], dim=1))                               # [1, 128, 32, 32, 32]                      
        out = self.decoder3(torch.cat([self.up3(out), t2], dim=1))                               # [1, 64, 64, 64, 64]
        out = self.decoder4(torch.cat([self.up4(out), t1], dim=1))                               # [1, 32, 128, 128, 128]

        # 输出层
        out = self.output_conv(out)
        out = self.soft(out)

        return out


class UNet3D_ResSE(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0.2, use_dropout:bool=True, ln_spatial_shape:list=[]):
        super(UNet3D_ResSE, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
        # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
        # 编码器
        self.encoder1 = nn.Sequential(
            ResCBR_3x3(in_channels, 32),
            SE_Block(32, 16),
        )
        self.encoder2 = nn.Sequential(
            CBR_Block_3x3(32, 64),
            CBR_Block_3x3(64, 64),
        )
        self.encoder3 = nn.Sequential(
            CBR_Block_3x3(64, 128),
            CBR_Block_3x3(128, 128),
        )
        self.encoder4 = nn.Sequential(
            CBR_Block_3x3(128, 256),
            CBR_Block_3x3(256, 256),
        )
        self.encoder5 = nn.Sequential(
            CBR_Block_3x3(256, 512),
            CBR_Block_3x3(512, 512),
        )
        # 解码器
        self.decoder1 = nn.Sequential(
            CBR_Block_3x3(512, 256),
            CBR_Block_3x3(256, 256),
        )
        self.up1      = nn.Sequential(
            Up_Block(512, 512, 4, 2, 1),
            CBR_Block_3x3(512, 256),
        )

        self.decoder2 = nn.Sequential(
            CBR_Block_3x3(256, 128),
            CBR_Block_3x3(128, 128),
        )
        self.up2      = nn.Sequential(
            Up_Block(256, 256, 4, 2, 1),
            CBR_Block_3x3(256, 128),
        )

        self.decoder3 = nn.Sequential(
            CBR_Block_3x3(128, 64),
            CBR_Block_3x3(64, 64),
        )
        self.up3      = nn.Sequential(
            Up_Block(128, 64, 4, 2, 1),
            CBR_Block_3x3(64, 64),
        )

        self.decoder4 = nn.Sequential(
            CBR_Block_3x3(64, 32),
            CBR_Block_3x3(32, 32),
        )
        self.up4      = nn.Sequential(
            Up_Block(64, 32, 4, 2, 1),
            CBR_Block_3x3(32, 32),
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

        # out = self.dropout(out) if self.use_dropout else out
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

    from ref.CBR_Blocks import *
    from ref.modules import Up_Block
    from attentions.SE import SE_Block
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = UNet3D_BN(in_channels=4, out_channels=4, use_dropout=False)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D_BN(in_channels=4, out_channels=4)
    input = torch.rand((1, 4, 128, 128, 128)).to(device)
    model = model.to(device)
    print(model)
    output = model(input)
    print(output.shape)
    summary(model, (1, 4, 128, 128, 128))