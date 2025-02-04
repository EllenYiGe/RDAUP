import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordinateAttention(nn.Module):
    """
    Coordinate Attention Module (Reference: Coordinate Attention, Hou et al. 2021):
    - Perform global pooling along the H and W dimensions, then concatenate along the channel dimension.
    - Generate attention weights along the H and W directions, and apply them to the features.
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        x: shape (B, C, H, W)
        Returns:
          Output of the same size (B, C, H, W)
        """
        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x)  # (b,c,1,w)

        # Concatenate and apply convolution/BN/ReLU
        y = torch.cat([x_h, x_w], dim=2)  # (b,c,h+1,1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y)

        # Split back into separate components
        x_h_, x_w_ = torch.split(y, [h, w], dim=2)  # (b,c,h,1) / (b,c,w,1)
        x_h_ = self.conv_h(x_h_.contiguous())        # (b,outC,h,1)
        x_w_ = self.conv_w(x_w_.contiguous())        # (b,outC,w,1)

        sigma_h = torch.sigmoid(x_h_)
        sigma_w = torch.sigmoid(x_w_.transpose(2,3)) # (b,outC,1,w)

        out = x * sigma_h
        out = out * sigma_w
        return out
