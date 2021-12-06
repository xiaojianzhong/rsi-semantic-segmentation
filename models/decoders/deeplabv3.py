import torch.nn as nn

from models.modules import ASPP


class DeepLabV3Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(DeepLabV3Decoder, self).__init__()
        self.aspp = ASPP(in_channels, out_channels)
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.classifier = nn.Conv2d(out_channels, num_classes if num_classes > 2 else 1, 1)

    def forward(self, x):
        x = self.aspp(x)
        x = self.convbnrelu(x)
        x = self.classifier(x)
        return x
