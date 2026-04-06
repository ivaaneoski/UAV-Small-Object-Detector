"""
CBAM (Convolutional Block Attention Module) for UAV Small Object Detection.

CBAM applies two sequential attention operations:
  1. Channel Attention — reweights feature channels based on global context
     (AdaptiveAvgPool + AdaptiveMaxPool -> shared MLP -> sigmoid)
  2. Spatial Attention — highlights which spatial regions matter most
     (channel-wise avg+max pooling -> 7x7 conv -> sigmoid)

This module amplifies feature responses from small aerial objects while
suppressing background noise, directly addressing the challenge that small
objects occupy very few pixels in drone-captured imagery.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Computes attention weights for each channel by pooling spatial information
    (both average and max) and passing through a shared MLP. This answers
    "what features are important?" by learning which channels carry the most
    discriminative information for small object detection.

    Args:
        in_channels: Number of input feature channels (C in BxCxHxW).
        reduction: Reduction ratio for the MLP bottleneck (default 16).
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass: pool -> shared MLP -> combine -> sigmoid scale.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Channel-attention-weighted feature map of same shape.
        """
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Computes a spatial attention map by concatenating channel-wise average
    and max pooled features, then convolving with a 7x7 kernel. This answers
    "where in the image should we focus?" by learning spatial locations that
    correspond to object regions.

    Args:
        kernel_size: Convolution kernel size for spatial attention map (default 7).
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass: channel pooling -> concat -> conv -> sigmoid scale.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Spatial-attention-weighted feature map of same shape.
        """
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined)) * x


class CBAM(nn.Module):
    """
    Full CBAM: Channel Attention followed by Spatial Attention in sequence.

    The module takes an input feature map, first reweights its channels to
    amplify informative features, then reweights spatial locations to focus
    on object regions. The output has the same shape as the input but with
    small objects amplified and background suppressed.

    Architecture flow:
        Input Feature Map [B x C x H x W]
              |
              v
        Channel Attention ("What to focus on")
              |
              v
        Spatial Attention ("Where to focus")
              |
              v
        Attended Feature Map [B x C x H x W]

    Args:
        in_channels: Number of input feature channels.
        reduction: Reduction ratio for channel attention MLP (default 16).
        kernel_size: Kernel size for spatial attention conv (default 7).
    """

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Forward pass: channel attention -> spatial attention.

        Args:
            x: Input feature tensor of shape [B, C, H, W].

        Returns:
            Attended feature tensor of shape [B, C, H, W].
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
