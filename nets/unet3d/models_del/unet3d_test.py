import torch
from torchinfo import summary
import torch.nn as nn
from torch.nn import functional as F

# from nets.unet3d.ref.modules import Up_Block
# from nets.unet3d.attentions.SE import SE_Block
# from nets.unet3d.ref.CBR_Blocks import *

# class UNet3D_BN(nn.Module):
#     def __init__(self, 
#                  in_channels:int, 
#                  out_channels:int, 
#                  dropout_rate:float=0.2, 
#                  use_dropout:bool=True, 
#                  ln_spatial_shape:list=[],
#                  features = [32, 64, 128, 256]):
#         super(UNet3D_BN, self).__init__()
#         # self.dropout_rate = dropout_rate
#         # self.use_dropout = use_dropout
#         # self.encoder_use_list = (use_bn, use_ln, False, 0.1)
#         # self.decoder_use_list = (use_bn, use_ln, False, 0.1)
#         self.encoder_features = features
#         self.decoder_features = (features + [features[-1]*2])[::-1]
#         # self.decoder_features = features[::-1]
#         self.up_features = self.decoder_features
#         self.bottom_layer = nn.Sequential(
#             CBR_Block_3x3(self.encoder_features[-1], self.encoder_features[-1]*2),
#             CBR_Block_3x3(self.encoder_features[-1]*2, self.encoder_features[-1]*2)
#         )
#         # 编码器
#         self.encoders = nn.ModuleDict()
#         self.up_layers = nn.ModuleDict()
#         self.decoders = nn.ModuleDict()

#         # 构建编码器
#         for i in range(len(self.encoder_features)):
#             if i == 0:
#                 setattr(self.encoders, f'encoder{i}_to_{self.encoder_features[i]}', 
#                         nn.Sequential(
#                     CBR_Block_3x3(in_channels, self.encoder_features[i]),
#                     CBR_Block_3x3(self.encoder_features[i], self.encoder_features[i])
#                 ))
#             else:
#                 setattr(self.encoders, f'encoder{i}_to_{self.encoder_features[i]}', 
#                         nn.Sequential(
#                     CBR_Block_3x3(self.encoder_features[i-1], self.encoder_features[i]),
#                     CBR_Block_3x3(self.encoder_features[i],self. encoder_features[i])
#                 ))

#         # 构建解码器
#         for i in range(len(self.decoder_features)):
#             if i == len(self.decoder_features) -1:
#                 setattr(self.decoders, f'decoder{i}_to_{out_channels}', 
#                         nn.Conv3d(self.decoder_features[i], out_channels, kernel_size=1))
#             else:
#                 setattr(self.decoders, f'decoder{i}_to_{self.decoder_features[i+1]}', 
#                         nn.Sequential(
#                             CBR_Block_3x3(self.decoder_features[i], self.decoder_features[i+1]),
#                             CBR_Block_3x3(self.decoder_features[i+1], self.decoder_features[i+1])
#                         ))
    
#         for i in range(len(self.up_features)-1):
#             setattr(self.up_layers, f'up{i}_to_{self.up_features[i+1]}', 
#                     nn.Sequential(
#                         Up_Block(self.up_features[i], self.up_features[i], 4, 2, 1),
#                         CBR_Block_3x3(self.up_features[i], self.up_features[i+1])))
        

            

#         # 输出层
#         self.output_conv = nn.Conv3d(32, out_channels, kernel_size=1)

#         # 归一化层
#         self.dropout = nn.Dropout3d(dropout_rate)

#         self.soft = nn.Softmax(dim=1)
#         print(self.encoders)
#         print(self.up_layers)
#         print(self.decoders)
    
#     def forward(self, x):
#         skip_out_list = []
#         # 编码器
#         for i, (module_name, module) in enumerate(self.encoders.items()):
#             if i == 0:                
#                 skip_out = module(x) 
#                 print(f"encoder module name: {module_name}")
#                 print(skip_out.shape)
#             else:
#                 skip_out = module(F.max_pool3d(skip_out, 2, 2))
#                 print(f"encoder module name: {module_name}")
#                 print(skip_out.shape)
#             skip_out_list.append(skip_out)
#             out = skip_out
        
#         # bottom layers
#         out = self.bottom_layer(out)
#         out = F.max_pool3d(out, 2, 2)

#         # 解码器
#         for i, ((d_module_name, d_module), (up_module_name, up_module)) in enumerate(zip(self.decoders.items(), self.up_layers.items())):
#             if i < len(self.decoder_features):
#                 out = d_module(torch.cat([up_module(out), skip_out_list.pop()], dim=1))
#                 print(d_module_name)
#                 print(up_module_name)
#                 print(out.shape)
    

#         return out
    
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
        self.output_conv = nn.Conv3d(32, out_channels, kernel_size=1)

        # 归一化层
        self.dropout = nn.Dropout3d(dropout_rate)

        self.soft = nn.Softmax(dim=1)
        print(self.encoders)
        print(self.up_layers)
        print(self.decoders)
    
    def forward(self, x):
        skip_out_list = []
        # 编码器
        for i, (module_name, module) in enumerate(self.encoders.items()):
            if i == 0:                
                skip_out = module(x) 
                print(f"encoder module name: {module_name}")
                print(skip_out.shape)
            else:
                skip_out = module(F.max_pool3d(skip_out, 2, 2))
                print(f"encoder module name: {module_name}")
                print(skip_out.shape)
            skip_out_list.append(skip_out)
            out = skip_out
        
        # bottom layers
        out = self.bottom_layer(out)
        out = F.max_pool3d(out, 2, 2)

        # 解码器
        for i, ((d_module_name, d_module), (up_module_name, up_module)) in enumerate(zip(self.decoders.items(), self.up_layers.items())):
            if i < len(self.decoder_features):
                out = d_module(torch.cat([up_module(out), skip_out_list.pop()], dim=1))
                print(d_module_name)
                print(up_module_name)
                print(out.shape)
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