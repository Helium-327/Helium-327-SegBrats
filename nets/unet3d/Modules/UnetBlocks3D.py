# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/21 20:57:56
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: unet3d的构建基础模块
=================================================
'''


import torch
import torch.nn as nn
from torch.nn import functional as F
# from Attentions3D import *
# from Modules.Attentions3D import *
# from nets.unet3d.Modules.Attentions3D import *

class CBR_Block_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CBR_Block_3x3, self).__init__()
        self.cbr_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.cbr_conv(x)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sample = nn.MaxPool3d(kernel_size=4, stride=2, padding=1)
    def forward(self, x):
        return self.down_sample(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up_sample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
                )
        else:
            self.up_sample = nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1)
                )
        
    def forward(self, x):
        return self.up_sample(x)
    
class ResAttCBR_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, att=False):
        super().__init__()
        block_list = []
        block_list.append(CBR_Block_3x3(in_channels, out_channels))
        if att:
            block_list.append(CBAM(out_channels))
        block_list.append(CBR_Block_3x3(out_channels, out_channels))
        self.double_cbr = nn.Sequential(*block_list)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return F.relu(self.conv(x) + self.double_cbr(x))


class Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch5 = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.out_conv = nn.Conv3d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ], dim=1)
        out = F.relu(self.out_conv(out))
        return out

class D_Inception_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        )
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=2, padding=2), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, dilation=3, padding=3), # 膨胀卷积，膨胀率为2
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
        )
        self.branch5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
        )
        self.out_conv = nn.Conv3d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        out = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ], dim=1)
        out = F.relu(self.out_conv(out))
        return out

class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential( 
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual_out = self.residual(x)
        shortcut_out = self.shortcut(x)
        out = self.relu(residual_out + shortcut_out)
        return self.downsample(out)
    

class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, upsample=True, dilation=False):
        super().__init__()
        if upsample:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear')
        else:
            self.upsample = nn.ConvTranspose3d(in_channels//2, in_channels//2, kernel_size=4, stride=2, padding=1)
        
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
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=1),
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
            assert x.shape == skipped.shape, "Skipped tensor must be provided for DecoderBottleneck"
            x = torch.cat([x, skipped], dim=1)
            
        residual_out = self.residual(x)
        shortcut_out = self.shortcut(x)
        out = self.relu(residual_out + shortcut_out)
        return out
    

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=4, num_hidden_layers=1):
        super(MLP, self).__init__()
        hidden_dim = in_channels // reduction_ratio
        hidden_layers_list = [nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1), nn.ReLU(inplace=True)] * num_hidden_layers
        self.input_layer = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.hidden_layers = nn.Sequential(*hidden_layers_list)
        self.output_layer = nn.Conv3d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x
    
class CAM(nn.Module):
    def __init__(self, in_dim, ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = MLP(in_dim, in_dim, ratio)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avgout = self.fc(self.avg_pool(x))
        maxout = self.fc(self.max_pool(x))
        out = avgout + maxout
        return self.sigmoid(out)

class FusionMagic(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2): # 32, 128
        super().__init__()
        # 分别对后两层的输入进行平均池化操作，得到每个通道的平均值
        self.avgpooloing = nn.AdaptiveAvgPool3d(1)
        self.maxpooling = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm([in_channels, 1, 1, 1])
        self.layer_norm2 = nn.LayerNorm([in_channels*2, 1, 1, 1])

        # 使用SE_Block进行将cat之后的特征进行压缩激发
        # self.SE_layer1 = SE_Block(in_channels*) # in_channels*6 = in)_channels*2 + in_channels*2*2
        self.layer_norm3 = nn.LayerNorm([in_channels*4, 1, 1, 1])

        self.layer_norm4 = nn.LayerNorm([in_channels*6, 1, 1, 1])

        self.layer_norm5 = nn.LayerNorm([in_channels*7, 1, 1, 1])

        self.MLP = nn.Sequential(
            nn.Conv3d(in_channels=in_channels*7, out_channels=in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels*7, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.Conv1 = nn.Conv3d(in_channels*7, out_channels, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs:list[torch.tensor]):
        # 对后两层的输入进行平均池化操作，得到每个通道的平均值
        x1 = self.avgpooloing(inputs[-1])
        x1 = self.maxpooling(x1)
        
        x1 = self.layer_norm1(x1)

        x2 = self.avgpooloing(inputs[-2])
        x2 = self.maxpooling(x2)
        x2 = self.layer_norm2(x2)

        x3 = self.avgpooloing(inputs[0])
        x3 = self.maxpooling(x3)
        x3 = self.layer_norm3(x3)

        out = torch.cat([x2, x3], dim=1)
        out = self.avgpooloing(out)
        out = self.maxpooling(out)
        out = self.layer_norm4(out)

        out = torch.cat([x1, out], dim=1)
        out = self.avgpooloing(out)
        out = self.maxpooling(out)
        out = self.layer_norm5(out)

        # out = self.dropout(out)

        out = self.MLP(out)
        out = self.Conv1(out)
        # out = self.SE_layer1(out)

        out = self.sigmoid(out)

        return out

class FusionMagic_v2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2): # 输入128， 输出32
        super(FusionMagic_v2, self).__init__()
        # 分别对后两层的输入进行平均池化操作，得到每个通道的平均值
        self.avgpooloing = nn.AdaptiveAvgPool3d(1)
        self.maxpooling = nn.AdaptiveMaxPool3d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.upsample_1 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='trilinear')
        # 使用SE_Block进行将cat之后的特征进行压缩激发
        # self.SE_layer1 = SE_Block(in_channels*) # in_channels*6 = in)_channels*2 + in_channels*2*2

        self.cam_128 = CAM(in_channels*4)

        self.cam_64 = CAM(in_channels*2)

        self.cam_32 = CAM(in_channels)

        self.cam_out = CAM(in_channels)

        self.cam_192 = CAM(in_channels*6)

        self.mlp_192 = MLP(in_channels=in_channels*6, out_channels=in_channels*4)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=(in_channels*4, 1, 1, 1))

        self.mlp_96 = MLP(in_channels=in_channels*3, out_channels=in_channels*2)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=(in_channels*2, 1, 1, 1))

        # self.mlp_288 = MLP(in_channels=in_channels*9, out_channels=in_channels*9)
        # self.layer_norm3 = nn.LayerNorm(normalized_shape=(in_channels*9, 1, 1, 1))

        self.cbr = nn.Sequential(
            CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels*6),
            CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels)
        )
            

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs:list[torch.tensor], pooling_type='all'):
        # 对后两层的输入进行平均池化操作，得到每个通道的平均值
        if pooling_type == 'all':
            x1 = self.cam_128(inputs[0])
            x2 = self.cam_64(inputs[1])
            x3 = self.cam_32(inputs[2])
        elif pooling_type == 'max':
            x1 = self.maxpooling(inputs[0])
            x2 = self.maxpooling(inputs[1])
            x3 = self.maxpooling(inputs[2])
        elif pooling_type == 'avg':
            x1 = self.avgpooloing(inputs[0])
            x2 = self.avgpooloing(inputs[1])
            x3 = self.avgpooloing(inputs[2])
        else:
            raise ValueError("Invalid pooling type")
        
        out1 = torch.cat([x1, x2], dim=1) # 128 + 64 = 192
        out1 = self.mlp_192(out1)
        out1 = self.layer_norm1(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.expand_as(inputs[0]) * inputs[0]

        out2 = torch.cat([x2, x3], dim=1) # 64 + 32 = 96
        out2 = self.mlp_96(out2)
        out2 = self.layer_norm2(out2)
        out2 = self.sigmoid(out2)
        out2 = out2.expand_as(inputs[1]) * inputs[1]

        out1 = self.upsample_1(out1) # 128 x 64
        out2 = self.upsample_2(out2) # 64 x 64

        out = torch.cat([out1, out2], dim=1) # 196
        out = self.cbr(out)
        out = self.cam_out(out)
        out = out.expand_as(inputs[2]) * inputs[2]
        return out
    
# class Adaptive_FusionMagic_v2(nn.Module):
#     def __init__(self, features, dropout=0.2): # 三输入为[128, 64, 32]; 四输入为 [256, 128, 64, 32]
#         super(FusionMagic_v2, self).__init__()
#         # 分别对后两层的输入进行平均池化操作，得到每个通道的平均值
#         self.avgpooloing = nn.AdaptiveAvgPool3d(1)
#         self.maxpooling = nn.AdaptiveMaxPool3d(1)
#         self.dropout = nn.Dropout(p=dropout)
#         self.upsample_1 = nn.Upsample(scale_factor=4, mode='trilinear')
#         self.upsample_2 = nn.Upsample(scale_factor=2, mode='trilinear')
#         # 使用SE_Block进行将cat之后的特征进行压缩激发
#         # self.SE_layer1 = SE_Block(in_channels*) # in_channels*6 = in)_channels*2 + in_channels*2*2


#         self.cams = [CAM(feature) for feature in features]
#         if len(features) == 3:
#             self.cams.append(CAM(features[-1]))
#             self.cams.append(CAM(features[-1] + features[0]))

#         self.mlp_192 = MLP(in_channels=in_channels*6, out_channels=in_channels*4)
#         self.layer_norm1 = nn.LayerNorm(normalized_shape=(in_channels*4, 1, 1, 1))

#         self.mlp_96 = MLP(in_channels=in_channels*3, out_channels=in_channels*2)
#         self.layer_norm2 = nn.LayerNorm(normalized_shape=(in_channels*2, 1, 1, 1))

#         # self.mlp_288 = MLP(in_channels=in_channels*9, out_channels=in_channels*9)
#         # self.layer_norm3 = nn.LayerNorm(normalized_shape=(in_channels*9, 1, 1, 1))

#         self.cbr = nn.Sequential(
#             CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels*6),
#             CBR_Block_3x3(in_channels=in_channels*6, out_channels=in_channels)
#         )
            

#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, inputs:list[torch.tensor], pooling_type='all'):
#         # 对后两层的输入进行平均池化操作，得到每个通道的平均值
#         if pooling_type == 'all':
#             x1 = self.cam_128(inputs[0])
#             x2 = self.cam_64(inputs[1])
#             x3 = self.cam_32(inputs[2])
#         elif pooling_type == 'max':
#             x1 = self.maxpooling(inputs[0])
#             x2 = self.maxpooling(inputs[1])
#             x3 = self.maxpooling(inputs[2])
#         elif pooling_type == 'avg':
#             x1 = self.avgpooloing(inputs[0])
#             x2 = self.avgpooloing(inputs[1])
#             x3 = self.avgpooloing(inputs[2])
#         else:
#             raise ValueError("Invalid pooling type")
        
#         out1 = torch.cat([x1, x2], dim=1) # 128 + 64 = 192
#         out1 = self.mlp_192(out1)
#         out1 = self.layer_norm1(out1)
#         out1 = self.sigmoid(out1)
#         out1 = out1.expand_as(inputs[0]) * inputs[0]

#         out2 = torch.cat([x2, x3], dim=1) # 64 + 32 = 96
#         out2 = self.mlp_96(out2)
#         out2 = self.layer_norm2(out2)
#         out2 = self.sigmoid(out2)
#         out2 = out2.expand_as(inputs[1]) * inputs[1]

#         out1 = self.upsample_1(out1) # 128 x 64
#         out2 = self.upsample_2(out2) # 64 x 64

#         out = torch.cat([out1, out2], dim=1) # 196
#         out = self.cbr(out)
#         out = self.cam_out(out)
#         out = out.expand_as(inputs[2]) * inputs[2]
#         return out

if __name__ == '__main__':
    from Modules.Attentions3D import *
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionMagic(32, 128).to(device)
    inputs = [torch.randn(1, 32, 128, 128, 128).to(device), torch.randn(1, 64, 64, 64, 64).to(device), torch.randn(1, 128, 32, 32, 32).to(device)]
    # print(model)
    output = model(inputs)
    print(output.shape)


        


        
        
        