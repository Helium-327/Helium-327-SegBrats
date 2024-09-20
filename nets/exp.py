import torch
import torch.nn as nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
         )
    def forward(self, x):
        return self.conv(x)

class UNet3d_bn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3d_bn, self).__init__()
        self.encoder1 = DoubleConv(in_channels, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder4 = DoubleConv(128, 256)
        self.encoder5 = DoubleConv(256, 512) 

        self.decoder1 = DoubleConv(512, 256)
        self.conv_trans1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.conv_trans2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(128, 64)
        self.conv_trans3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(64, 32)
        self.conv_trans4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.out_conv = DoubleConv(32, out_channels)

        self.soft = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        # 编码器部分
        t1 = self.encoder1(x)                                               # 32 x 128 x 128 x 128
        out = F.max_pool3d(t1, 2, 2)                                        # 32 x 64 x 64 x 64
                                    
        t2 = self.encoder2(out)                                             # 64 x 64 x 64 x 64
        out = F.max_pool3d(t2, 2, 2)                                        # 64 x 32 x 32 x 32
        
        t3 = self.encoder3(out)                                             # 128 x 32 x 32 x 32
        out = F.max_pool3d(t3, 2, 2)                                        # 128 x 16 x 16 x 16
        
        t4 = self.encoder4(out)                                             # 256 x 16 x 16 x 16
        out = F.max_pool3d(t4, 2, 2)                                        # 256 x 8 x 8 x 8
        
        out = self.encoder5(out)                                            # 512 x 8 x 8 x 8
        
        
        
        out = self.conv_trans1(out)                                         # 256 x 16 x 16 x 16
        out = self.decoder1(torch.cat([out, t4], dim=1))                    # 256 x 16 x 16 x 16
        
        out = self.conv_trans2(out)                                          # 128 x 32 x 32 x 32
        out = self.decoder2(torch.cat([out, t3], dim=1))                    # 128 x 32 x 32 x 32
        
        out = self.conv_trans3(out)                                         # 64 x 64 x 64 x 64
        out = self.decoder3(torch.cat([out, t2], dim=1))                    # 64 x 64 x 64 x 64                

        out = self.conv_trans4(out)                                         # 32 x 128 x 128 x 128
        out = self.decoder4(torch.cat([out, t1], dim=1))                    # 32 x 128 x 128 x 128

        out = self.out_conv(out)                                            # out_channels x 128 x 128
        
        out = self.soft(out)                                             # softmax
        return out
    

if __name__ == "__main__":
    # test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.randn([1, 4, 128, 128, 128]).float().to(device)
    # input_tensor.shape
    model = UNet3d_bn(4, 4).to(device)
    out = model(input_tensor)

    print(out.shape)