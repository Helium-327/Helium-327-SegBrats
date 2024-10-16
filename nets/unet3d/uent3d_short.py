# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/14 18:10:08
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: 尝试构建不同深度的unet3d模型
=================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ref.CBR_Blocks import CBR_Block_3x3, ResCBR_3x3
from ref.modules import Up_Block

class Encoder_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, d_rate, layer_depth=2):
        super().__init__()
        self.encoder_layers = _make_encoder_layer(in_channels, mid_channels, d_rate, layer_depth)
        # self.decoder_layers = _make_decoder_layer(mid_channels*d_rate*layer_depth, mid_channels*d_rate*layer_depth//d_rate, d_rate, layer_depth)
        
        # self.out_conv = nn.Conv3d(mid_channels, num_classes, 1, 1, 0)

        # self.soft = nn.Softmax(dim=1)
    def forward(self, x):
        skip_out = []
        for i in range(len(self.encoder_layers)):
            if i == 0:
                x = self.encoder_layers[i](x)
            else:
                x = self.encoder_layers[i](F.adaptive_max_pool3d(x, x.shape[-1]//2))
                # print(f"{i}th encoder layer shape is {x.shape}")

            skip_out.append(x)
            print(f"{i}th encoder layer shape is {skip_out[-1].shape}")
        return x, skip_out

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, d_rate, layer_depth=2):
        super().__init__()
        self.decoder_layers = _make_decoder_layer(in_channels, mid_channels, d_rate, layer_depth)
    
    def forward(self, x, skip_connections):
        for i in range(len(self.decoder_layers)-1, -1, -1):
            if i != 0:
                x = self.decoder_layers[i](torch.cat([x, skip_connections[i]], dim=1))
            else:
                x = self.decoder_layers[i](x)
         
def _make_encoder_layer(in_channels, mid_channels, d_rate, layer_depth=2):
    encoder_list = []
    for i in range(layer_depth):
        if i == 0:
            encoder_list.append(CBR_Block_3x3(in_channels, mid_channels))
        else:
            encoder_list.append(CBR_Block_3x3(mid_channels, mid_channels*d_rate))
            mid_channels *= d_rate

    encoder_layers = nn.Sequential(*encoder_list)

    return encoder_layers

def _make_decoder_layer(in_channels, mid_channels, d_rate, layer_depth=2):
    decoder_list = []
    for i in range(layer_depth):
        if i == 0:
            decoder_list.append(CBR_Block_3x3(in_channels, mid_channels))
        else:
            decoder_list.append(CBR_Block_3x3(mid_channels, mid_channels//d_rate))
            mid_channels //= d_rate

    decoder_layers = nn.Sequential(*decoder_list)

    return decoder_layers

def _clone_layer(layer, num):
    return nn.Sequential(*[layer for _ in range(num)])

if __name__ == '__main__':
    # layers = _clone_layer(nn.Conv3d(1, 1, 3, 1, 1), 3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # uncoder_layers = _make_encoder_layer(4, 32, 2, 4)
    # decoder_layers = _make_decoder_layer(512, 256, 2, 4)

    input_data = torch.randn(1, 4, 128, 128, 128).to(device=device)

    net = Encoder_Block(4, 32, d_rate=2, layer_depth=4).to(device=device)
    output, _ = net(input_data)

    print(output.shape)