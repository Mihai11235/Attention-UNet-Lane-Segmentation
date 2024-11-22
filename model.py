# model.py

import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # g: gating signal from the decoder
        # x: feature map from the encoder
        g = self.W_g(g)
        x = self.W_x(x)

        # Combine the two signals
        attn = self.relu(g + x)
        attn = self.psi(attn)
        attn = self.sigmoid(attn)

        # Apply the attention map to the feature map x
        return x * attn


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetWithAttention, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.attn4 = AttentionGate(512, 512, 256)  # Attention Gate for the skip connection
        self.dec4 = CBR(768, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(256, 256, 128)  # Attention Gate for the skip connection
        self.dec3 = CBR(384, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(128, 128, 64)  # Attention Gate for the skip connection
        self.dec2 = CBR(192, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(64, 64, 32)  # Attention Gate for the skip connection
        self.dec1 = CBR(96, 64)

        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoding path
        # dec4
        dec4 = self.upconv4(bottleneck)
        attn4 = self.attn4.forward(enc4, dec4)  # Apply attention gate
        dec4 = torch.cat((attn4, dec4), dim=1)  # Concatenate attention output and upsampled feature map
        dec4 = self.dec4(dec4)  # Adjust the number of input channels expected here

        # dec3
        dec3 = self.upconv3(dec4)
        attn3 = self.attn3.forward(enc3, dec3)  # Apply attention gate
        dec3 = torch.cat((attn3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        # dec2
        dec2 = self.upconv2(dec3)
        attn2 = self.attn2.forward(enc2, dec2)  # Apply attention gate
        dec2 = torch.cat((attn2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        # dec1
        dec1 = self.upconv1(dec2)
        attn1 = self.attn1.forward(enc1, dec1)  # Apply attention gate
        dec1 = torch.cat((attn1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.conv(dec1))

def create_unet(in_channels=3, out_channels=1):
    return UNetWithAttention(in_channels, out_channels)