

# class Encoder(nn.Module):
#     def __init__(self, in_channels=4, features=[16, 32, 64, 128, 256]):
#         super().__init__()
#         self.encoders = nn.ModuleList()
#         self.DownSample = nn.MaxPool3d(kernel_size=2, stride=2)

#         for feature in features:
#             self.encoders.append(EncoderBottleneck(in_channels, feature))
#             in_channels = feature

#     def forward(self, x):
#         skip_connections = []
#         for encoder in self.encoders:
#             x = encoder(x)
#             print(f"x: {x.shape}")
#             skip_connections.append(x)
#             print(f"skip:{x.shape}")
#             x = self.DownSample(x)
#         out = x
#         return out, skip_connections        

# class Decoder(nn.Module):
#     def __init__(self, out_channels, features=[256, 128, 64, 32, 16], upsample=False):
#         super().__init__()

#         self.decoders = nn.ModuleList()
#         for i, feature in enumerate(features):
#             if upsample:
#                 self.decoders.append(nn.Upsample(scale_factor=2, mode='trilinear'))
#             else:
#                 self.decoders.append(nn.ConvTranspose3d(feature, feature, kernel_size=4, stride=2, padding=1))

#             if i == len(features)-1:
#                 self.decoders.append(DecoderBottleneck(feature*2, out_channels, scale_factor=2, upsample=upsample, dilation=True))
#             else:
#                 self.decoders.append(DecoderBottleneck(feature*2, feature//2, scale_factor=2, upsample=upsample))


#     def forward(self, x, skip_connections):
#         skip_connections = skip_connections[::-1]
#         decoder_out_list = []
#         for decoder in self.decoders:
#             print(f"skip: {skip_connections[-1].shape}")
#             print(f"x: {x.shape}")
#             if isinstance(decoder, DecoderBottleneck):
#                 x = decoder(x, skip_connections.pop())
#                 decoder_out_list.append(x)
#             elif isinstance(decoder, nn.ConvTranspose3d | nn.Upsample):
#                 x = decoder(x)
#         return x