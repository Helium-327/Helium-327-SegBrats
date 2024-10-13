# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/09 12:14:52
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: self-attention
=================================================
'''
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, in_channels, dk, dv):
        super(SelfAttention, self).__init__()
        self.scale = dk ** -0.5
        self.q = nn.Linear(in_channels, dk)
        self.k = nn.Linear(in_channels, dk)
        self.v = nn.Linear(in_channels, dv)

    def forward(self, x):
        batch_size, num_patches, in_channels = x.size()

        q = self.q(x).view(batch_size, num_patches, -1)
        k = self.k(x).view(batch_size, num_patches, -1)
        v = self.v(x).view(batch_size, num_patches, -1)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        x = attention @ v
        x = x.view(batch_size, num_patches, -1)

        return x

if __name__ == '__main__':
    att = SelfAttention(dim=128, dk=128, dv=4)

    x = torch.rand((1, 4, 128*128*128))
    output = att(x)

    print(output.shape)