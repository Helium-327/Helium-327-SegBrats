import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (128 // patch_size) ** 3  # Calculate the number of patches

    def forward(self, x):
        x = self.proj(x)  # (B, C, D, H, W) -> (B, E, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, E, N)
        return x

class PositionalEncoding3D(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding3D, self).__init__()
        self.embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.embedding

class EncoderLayer3D(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(EncoderLayer3D, self).__init__()
        self.self_attn = nn.MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.self_attn(x, x)[0]
        x = x + self.mlp(self.norm1(x))
        return self.norm2(x)

class DecoderLayer3D(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(DecoderLayer3D, self).__init__()
        self.self_attn = nn.MultiHeadAttention(embed_dim, num_heads)
        self.encoder_attn = nn.MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_output):
        x = x + self.self_attn(x, x)[0]
        x = x + self.encoder_attn(x, encoder_output)[0]
        x = x + self.mlp(self.norm1(x))
        return self.norm3(x)

class ViT3DSegmentation(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, mlp_ratio, num_layers, num_classes):
        super(ViT3DSegmentation, self).__init__()
        self.patch_embed = PatchEmbedding3D(in_channels, embed_dim, 8)
        self.pos_embed = PositionalEncoding3D(self.patch_embed.num_patches, embed_dim)
        self.encoder_layers = nn.ModuleList([EncoderLayer3D(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer3D(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for layer in self.encoder_layers:
            x = layer(x)
        for layer in self.decoder_layers:
            x = layer(x, x)  # Decoder uses the same features for simplicity
        x = self.mlp_head(x[:, 0])
        return x

# Model parameters
in_channels = 1
embed_dim = 768
patch_size = 8
num_heads = 12
mlp_ratio = 4
num_layers = 6
num_classes = 10

# Create the model
model = ViT3DSegmentation(in_channels, embed_dim, num_heads, mlp_ratio, num_layers, num_classes)

# Example input
input_tensor = torch.randn(1, in_channels, 128, 128, 128)

# Forward pass
output = model(input_tensor)
print(output.shape)  # Should be (1, num_classes)