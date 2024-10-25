# -*- coding: UTF-8 -*-
'''
================================================
*      CREATE ON: 2024/10/24 15:44:02
*      AUTHOR: @Junyin Xiong
*      DESCRIPTION: Vision Transformer
=================================================
'''
import torch
import torch.nn as nn

from torch.nn import functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channnels, patch_size, embed_dim, num_patches, dropout):
        super(PatchEmbedding, self).__init__()

        self.patcher = nn.Sequential(
            nn.Conv3d(in_channels=in_channnels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),

        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim), requires_grad=True)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # 对齐batch_size

        x = self.patcher(x).permute(0, 2, 1)  # 切块
        x = torch.cat([x, cls_token], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)

        return x


class VitBlock(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patched, dropout,
                num_heads, activation, num_encoders, num_classes): 
        super(VitBlock, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patched, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim, dropout=dropout, 
                                                   activation=activation,
                                                   batch_first=True,
                                                   norm_first=False)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)
        # x = self.MLP(x[:, 0, :])
        x = x.view(1, 4, 128, 128, 128)

        upsampled_x = F.interpolate(x, size=(128, 128, 128), mode='trilinear', align_corners=False)
        return upsampled_x
    
if __name__ == '__main__':
    in_channels =  4
    patch_size = 8
    embed_dim = 768
    num_pathes = 4096
    dropout = 0.1
    num_heads = 12
    activation = nn.GELU()
    num_encoders = 4
    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit = VitBlock(in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim, num_patched=num_pathes, dropout=dropout,
                    num_heads=num_heads, activation=activation, num_encoders=num_encoders, num_classes=num_classes)
    vit = vit.to(device)
    print(vit)
    input_data = torch.randn(1, in_channels, 128, 128, 128)
    input_data = input_data.to(device)
    output = vit(input_data)
    print(output.shape)