# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/15 16:23:27
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: PSPNet
=================================================
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPNET(nn.Module):
    def __init__(self,
                  in_channel, 
                  mid_channels, 
                  out_channels,
                  img_size,
                  op_conv, 
                  op_bn, 
                  op_act, 
                  depth,
    ):
        super(PSPNET, self).__init__()   
        self.in_channel = in_channel
        self.mid_channel = mid_channels   
        self.out_channel = out_channels
        self.img_size = img_size           
        self.op_conv = op_conv
        self.op_bn = op_bn
        self.op_act = op_act
        self.depth = depth

        self.pool = nn.AdaptiveAvgPool3d
        self.up_sampling = nn.ConvTranspose3d
        self.psp_layers = nn.Sequential(*self._make_pooling_layer())
        self.fe_layers = nn.Sequential(*self._make_Feature_Extraction_layers())
        self.up_layers = nn.Sequential(*self._make_up_layers())
    
    def _make_pooling_layer(self):
        
        pooling_layers = []
        for i in range(self.depth):
            if i == 0:
                pooling_layers.append(
                    nn.Sequential(
                            self.pool(self.img_size//(2*self.depth)),
                            self.op_conv(self.in_channel, self.mid_channel, kernel_size=1),
                            self.op_bn(self.mid_channel),
                            self.op_act()))
            else:
                pooling_layers.append(
                    nn.Sequential(
                            self.pool(self.img_size//(2*self.depth)),
                            self.op_conv(self.in_channel, self.mid_channel*(2**i), kernel_size=1),
                            self.op_bn(self.mid_channel*(2**i)),
                            self.op_act()))
                
        return pooling_layers

    def _make_Feature_Extraction_layers(self, dilation=1, padding=1):
        fe_layers = []
        for i in range(self.depth):
            if i == 0:
                fe_layers.append(
                    nn.Sequential(
                        self.op_conv(self.mid_channel, self.mid_channel, kernel_size=3, dilation=dilation, padding=padding),
                        self.op_bn(self.mid_channel),
                        self.op_act(), 
                        self.op_conv(self.mid_channel, self.out_channel, kernel_size=1),
                        self.op_bn(self.out_channel),
                        self.op_act()))

            else:
                fe_layers.append(
                    nn.Sequential(
                        self.op_conv(self.mid_channel*(2**i), self.mid_channel*(2**i),  kernel_size=3, dilation=dilation, padding=padding),
                        self.op_bn(self.mid_channel*(2**i)),
                        self.op_act(),
                        self.op_conv(self.mid_channel*(2**i), self.out_channel, kernel_size=1),
                        self.op_bn(self.out_channel),
                        self.op_act()))
        return fe_layers
        
    def _make_up_layers(self):
        up_layers = []
        for i in range(self.depth):
            if i == 0:
                up_layers.append(
                nn.Sequential(
                    self.op_conv(self.out_channel*self.depth, self.out_channel, kernel_size=1, padding=1),
                    self.op_bn(self.out_channel),
                    self.op_act()))
            up_layers.append(
            nn.Sequential(
                self.up_sampling(self.out_channel//(2**(i-1)), self.out_channel//(2**(i)), kernel_size=2, stride=2),
                self.op_bn(self.out_channel),
                self.op_act()))
                # self.op_conv(self.out_channel*self.depth, self.out_channel*self.depth, kernel_size=3, padding=1),
                # self.op_bn(self.out_channel*self.depth),
                # self.op_act()))
        return up_layers


        
    def forward(self, x):
        pooling_output = []
        for m in self.psp_layers:
            out = m(x)
            pooling_output.append(out)
        fe_output = []
        for i, m in enumerate(self.fe_layers):
            out = m(pooling_output[i])
            fe_output.append(out)
        out = torch.cat(fe_output, dim=1)

        for m in self.up_layers:
            out = m(out)
    
        return pooling_output, fe_output, out
    



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = PSPNET(4, 128, 128, 128, nn.Conv3d, nn.BatchNorm3d, nn.ReLU, 4).to(device)
    input = torch.randn(1, 4, 32, 32, 32).to(device)

    pooling_output, fe_output, out = net(input)

    for output in pooling_output:
        print(output.shape)

    for output in fe_output:
        print(output.shape)
    
    print(out.shape)

                  