import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation):
        super(ASPPConv, self).__init__()
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.convbnrelu(x)
        return x


class ASPPPool(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ASPPPool, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.gap(x)
        x = self.convbnrelu(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=(12, 24, 36)):
        super(ASPP, self).__init__()
        self.branches = nn.ModuleList()
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
        for dilation in dilations:
            self.branches.append(ASPPConv(in_channels, out_channels, dilation))
        self.branches.append(ASPPPool(in_channels, out_channels))

        self.convbnrelu = nn.Sequential(
            nn.Conv2d(out_channels*(len(dilations)+2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        xs = []
        for branch in self.branches:
            xs.append(branch(x))
        x = torch.cat(xs, dim=1)
        x = self.convbnrelu(x)
        return x
