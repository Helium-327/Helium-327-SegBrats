from torch import nn
import torch



class UNet3D_original(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_p=0.2):
        super(UNet3D_original, self).__init__()
        # self.ker_init = nn.init.he_normal_
        self.dropout_p = dropout_p
        self.maxPooling = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.Conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.Conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.Conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        ) # 
        self.Conv5 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        ) # c = 512
        self.dropout = nn.Dropout(p=self.dropout_p)
        
        # TODO: 转置卷积怎么用
        # H_out = (H_in - 1) * stride - 2 * padding + kernel_size
        self.upSampling3d_1 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=4, stride=2, padding=1), # 上采样 8 ---> 16
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 256, kernel_size=3, padding=1), 
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        ) # c =256
        
        # 与Conv4的输出concate拼接，之后的 c= 512
        self.Conv6 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.upSampling3d_2  = nn.Sequential(
            nn.ConvTranspose3d(256, 256, kernel_size=4, stride=2, padding=1), # 上采样 16 ---> 32
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        # 与Conv3的输出concate拼接，之后的 c= 256
        self.Conv7 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.upSampling3d_3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1), # 上采样 32 ---> 64
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        # 与Conv2的输出concate拼接，之后的 c= 128
        
        self.Conv8 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upSampling3d_4 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, padding=1), # 上采样 64 ---> 128
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        # 与Conv1的输出concate拼接，之后的 c= 64
        self.Conv9 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.ConvOutput = nn.Conv3d(32, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, x):
        input_layer = self.Conv1(x)  # 2 x 128 x 128 ----> 32 x 128 x 128
        down1 = self.maxPooling(input_layer) # 32 x 64 x 64
        down2 = self.maxPooling(self.Conv2(down1)) # 64 x 32 x 32
        down3 = self.maxPooling(self.Conv3(down2)) # 128 x 16 x 16
        down4 = self.maxPooling(self.Conv4(down3)) # 256 x 8 x 8
        down_ouput = self.Conv5(down4) # 512 x 8 x 8
        
        if self.dropout_p > 0:
            down_ouput = self.dropout(down_ouput) # 512 x 8 x 8
        up1 = self.upSampling3d_1(down_ouput) # 256 x 16 x 16
        up1_cat_down4 = torch.cat([up1, self.Conv4(down3)], dim=1) # [256 x 16 x 16, 256 x 16 x 16] ----> 512 x 16 x 16
        up2 = self.Conv6(up1_cat_down4) # 256 x 32 x 32
        up3 = self.upSampling3d_2(up2) # 128 x 32 x 32
        up3_cat_down3 = torch.cat([up3, self.Conv3(down2)], dim=1) # [128 x 32 x 32, 128 x 32 x 32] ----> 256 x 32 x 32
        up4 = self.Conv7(up3_cat_down3) # 128 x 32 x 32
        up5 = self.upSampling3d_3(up4)  # 64 x 64 x 64
        up5_cat_down2 = torch.cat([up5, self.Conv2(down1)], dim=1) # [64 x 64 x 64, 64 x 64 x 64] ----> 128 x 64 x 64
        up6 = self.Conv8(up5_cat_down2) # 64 x 64 x 64
        up7 = self.upSampling3d_4(up6)  # 32 x 128 x 128
        up7_cat_down1 = torch.cat([up7, input_layer], dim=1) # [32 x 128 x 128, 32 x 128 x 128] ----> 64 x 128 x 128
        up8 = self.Conv9(up7_cat_down1) # 32 x 128 x 128
        output = self.ConvOutput(up8) # num_class x 128 x 128
        out = self.softmax(output)
        return out     