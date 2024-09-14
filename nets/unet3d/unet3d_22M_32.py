import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):        # FIXME: 初始化之后损失异常
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)
        self.down4 = DoubleConv(128, 256)
        self.down5 = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.up1_conv = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.up2_conv = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.up3_conv = DoubleConv(128, 64)
        
        self.up4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.up4_conv = DoubleConv(64, 32)

        self.out_conv = nn.Conv3d(32, out_channels, kernel_size=1)

        self.MaxPooling3d = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.initialize_weights(init_type="kaiming_normal", activation="relu")

    def forward(self, x):
        down1_out = self.down1(x)                                               # 64 x 224 x 224 x 224
        down2_out = self.down2(self.MaxPooling3d(down1_out))                    # 128 x 112 x 112 x 112
        down3_out = self.down3(self.MaxPooling3d(down2_out))                    # 256 x 56 x 56 x 56
        down4_out = self.down4(self.MaxPooling3d(down3_out))                    # 512 x 28 x 28 x 28
        down5_out = self.down5(self.MaxPooling3d(down4_out))                    # 1024 x 14 x 14 x 14

        up1_out = self.up1(down5_out)                                           # 512 x 28 x 28 x 28
        up1_cat_out = torch.cat([up1_out, down4_out], dim=1)                    # 1024 x 28 x 28 x 28
        up1_conv_out = self.up1_conv(up1_cat_out)                               # 512 x 28 x 28 x 28

        up2_out = self.up2(up1_conv_out)                                           # 256 x 56 x 56 x 56
        up2_cat_out = torch.cat([up2_out, down3_out], dim=1)                    # 512 x 56 x 56 x 56
        up2_conv_out = self.up2_conv(up2_cat_out)                               # 256 x 56 x 56 x 56

        up3_out = self.up3(up2_conv_out)                                           # 128 x 112 x 112 x 112
        up3_cat_out = torch.cat([up3_out, down2_out], dim=1)                    # 256 x 112 x 112 x 112
        up3_conv_out = self.up3_conv(up3_cat_out)                               # 128 x 112 x 112 x 112

        up4_out = self.up4(up3_conv_out)                                           # 64 x 224 x 224 x 224
        up4_cat_out = torch.cat([up4_out, down1_out], dim=1)                    # 128 x 224 x 224 x 224
        up4_conv_out = self.up4_conv(up4_cat_out)                               # 64 x 224 x 224 x 224

        out = self.out_conv(up4_conv_out)                                       # out_channel x 224 x 224 x 224
        
        out = self.softmax(out)
        return out
    
    def initialize_weights(self, init_type='normal', activation='relu', init_gain=0.02, always_init=True):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                if init_type == 'kaiming_normal':
                    if isinstance(m, nn.ConvTranspose3d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation)  # `fan_in` for ConvTranspose3d
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)  # `fan_out` for Conv3d
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0, init_gain)
                
                if always_init or not (init_type in ['kaiming_normal', 'xavier_normal']):
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm3d):
                if always_init or init_type in ['kaiming_normal', 'xavier_normal']:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                if init_type == 'kaiming_normal':
                    if isinstance(m, nn.ConvTranspose3d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation)
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                else:
                    nn.init.normal_(m.weight, 0, init_gain)
                    
                if always_init:
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)



    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet3D(4, 4)
    model.to(device)
    input_tensor = torch.randn(1, 4, 128, 128, 128)
    input_tensor = input_tensor.to(device)
    print(input_tensor.shape)
    out = model(input_tensor)
