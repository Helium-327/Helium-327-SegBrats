import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary
from nets.unet3d.ref.CBR_Blocks import *


class CLR_Block_3x3(CBR_Block_3x3):
    def __init__(self, in_channels:int, out_channels:int, ln_spatial_shape:list=[]):
        super(CLR_Block_3x3, self).__init__(in_channels, out_channels)
        # 参数
        self.conv[2] = nn.LayerNorm([out_channels, *ln_spatial_shape])
    
    def forward(self, x):
        out = self.conv(x)
        return out

class CLR_Block_5x5(CBR_Block_5x5):
    def __init__(self, in_channels:int, out_channels:int, ln_spatial_shape:list=[]):
        super(CLR_Block_5x5, self).__init__(in_channels, out_channels)
        # 参数
        self.conv[2] = nn.LayerNorm([out_channels, *ln_spatial_shape])

class CLR_Block_Dilation(CBR_Block_Dilation):
    def __init__(self, in_channels:int, out_channels:int, ln_spatial_shape:list=[]):
        super(CLR_Block_Dilation, self).__init__(in_channels, out_channels)
        # 参数
        self.conv[2] = nn.LayerNorm([out_channels, *ln_spatial_shape])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLR_Block_3x3(4, 32).to(device)

    summary(model, (4, 128, 128, 128))
    x = torch.rand((1, 4, 128, 128, 128)).to(device)
    out = model(x)
    print(out.shape)