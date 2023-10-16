
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_architecture import Up, ConvolutionalBlock, Down, OutConv

# Attention gate
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g_prime = self.W_g(g)
        x_prime = self.W_x(x)
        psi = F.relu(g_prime + x_prime)
        psi = self.psi(psi)
        return x * psi

# Attention UNet
class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet, self).__init__()

        self.encoder = nn.ModuleList([
            ConvolutionalBlock(in_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
        ])

        self.middle = ConvolutionalBlock(512, 1024)

        self.decoder = nn.ModuleList([
            Up(1024, 512),
            Up(512, 256),
            Up(256, 128),
            Up(128, 64),
        ])

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(512, 512),
            AttentionBlock(256, 256),
            AttentionBlock(128, 128),
            AttentionBlock(64, 64),
        ])

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for block in self.encoder:
            x = block(x)
            skip_connections.append(x)

        x = self.middle(x)

        # Decoder
        for up_conv, att_block, skip in zip(self.decoder, self.attention_blocks, reversed(skip_connections)):
            x = up_conv(x)
            x = att_block(x, skip)

        x = self.out_conv(x)

        return x