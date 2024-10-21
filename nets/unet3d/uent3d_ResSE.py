import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.unet3d.ref.modules import Up_Block
from nets.unet3d.attentions.SE import SE_Block
from nets.unet3d.ref.CBR_Blocks import *

from torchsummary import summary


class UNet3D_ResSE(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, dropout_rate:float=0.2, use_dropout:bool=True, ln_spatial_shape:list=[]):
        super(UNet3D_ResSE, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout
        # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
        # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
        # 编码器
        self.encoder1 = nn.Sequential(
            ResCBR_3x3(in_channels, 32)
        )
        self.attention_1 =  SE_Block(32, 32)

        self.encoder2 = nn.Sequential(
            ResCBR_3x3(32, 64)
        )
        self.attention_2 = SE_Block(64, 64)

        self.encoder3 = nn.Sequential(
            ResCBR_3x3(64, 128)
        )
        self.attention_3 = SE_Block(128, 128)

        self.encoder4 = nn.Sequential(
            ResCBR_3x3(128, 256)
        )
        self.attention_4 = SE_Block(256, 256)

        self.encoder5 = nn.Sequential(
            ResCBR_3x3(256, 512)
        )
        self.attention_5 = SE_Block(512, 512)

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
        t1 = self.encoder1(x)           
        t1 = self.attention_1(t1)                                                        # [1, 32, 128, 128, 128]
        t2 = self.encoder2(F.max_pool3d(t1, 2, 2))                                              # [1, 64, 64, 64, 64] 
        t2 = self.attention_2(t2)
        t3 = self.encoder3(F.max_pool3d(t2, 2, 2))                                              # [1, 128, 32, 32, 32]
        t3 = self.attention_3(t3)
        t4 = self.encoder4(F.max_pool3d(t3, 2, 2))                                              # [1, 256, 16, 16, 16]
        t4 = self.attention_4(t4)
        out = self.encoder5(F.max_pool3d(t4, 2, 2))                                             # [1, 512, 8, 8, 8]
        out = self.attention_5(out)

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
    model = UNet3D_ResSE(in_channels=4, out_channels=4, use_dropout=False)
    print(model)
    input_tensor = torch.randn([1, 4, 128, 128, 128]).float()

    model.to(device)
    input_tensor = input_tensor.to(device)


    out = model(input_tensor)
    summary(model, (4, 128, 128, 128))
    print(out.shape)