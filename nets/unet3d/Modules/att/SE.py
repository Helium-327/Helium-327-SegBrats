# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/11 21:46:07
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: SENet 和 SE注意力
=================================================
'''
import torch
import torch.nn as nn

class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SE_Block, self).__init__()

        if in_channels // reduction_ratio <= 0:
                    raise ValueError(f"Reduction ratio {reduction_ratio} is too large for the number of input channels {in_channels}.")
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
                # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y.expand_as(x)
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SE(64, 16).to(device)
    x = torch.rand((1, 64, 64, 64, 64)).to(device)
    out = model(x)
    print(out.shape)

