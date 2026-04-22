"""
ECA (Efficient Channel Attention) Module for UAV Small Object Detection.

ECA is a lightweight attention module that avoids dimensionality reduction 
and performs cross-channel interaction via a 1D convolution. 
It adaptively determines the kernel size of the 1D convolution based on 
the input channel dimension, allowing it to be lighter than CBAM while 
effectively amplifying informative channels for small object detection.

Source paper: WildYOLO (Vijaykumar et al., IIT Goa, ETAAV 2025)
"""

import math
import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    ECA (Efficient Channel Attention) Module.
    
    Computes channel attention using a 1D convolution without dimensionality 
    reduction, capturing local cross-channel interaction efficiently.
    
    Args:
        in_channels: Number of input feature channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Compute adaptive kernel size: k = nearest odd integer to (log2(in_channels) / 2 + 0.5)
        k = int(math.log2(in_channels) / 2.0 + 0.5)
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass: gap -> squeeze -> 1D conv -> sigmoid -> unsqueeze -> scale.

        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Channel-attention-weighted feature map of same shape.
        """
        b, c, _, _ = x.size()
        z = self.gap(x)           # [B, C, 1, 1]
        z = z.view(b, 1, c)       # [B, 1, C]
        attn = self.sigmoid(self.conv(z)) # [B, 1, C]
        attn = attn.view(b, c, 1, 1) # [B, C, 1, 1]
        return x * attn


if __name__ == '__main__':
    # Quick shape assertion test
    x_test = torch.randn(1, 64, 32, 32)
    eca_module = ECA(64)
    out_test = eca_module(x_test)
    assert out_test.shape == x_test.shape, "Output shape mismatch!"
    print("ECA module test passed!")
