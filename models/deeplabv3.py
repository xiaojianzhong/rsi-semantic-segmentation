import torch.nn as nn
import torch.nn.functional as F

from models.encoders import ResNet
from models.decoders import DeepLabV3Decoder
from models.utils.init import initialize_weights


class DeepLabV3ResNet(nn.Module):
    def __init__(self, depth, in_channels, num_classes):
        super(DeepLabV3ResNet, self).__init__()
        self.encoder = ResNet(depth, in_channels)
        depth2channels = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
        }
        self.decoder = DeepLabV3Decoder(depth2channels[depth], 512, num_classes)
        initialize_weights(self.decoder)

    def forward(self, x):
        _, _, h, w = x.shape

        x = self.encoder(x)[-1]
        x = self.decoder(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class DeepLabV3ResNet18(DeepLabV3ResNet):
    def __init__(self, num_channels, num_classes):
        super(DeepLabV3ResNet18, self).__init__(18, num_channels, num_classes)


class DeepLabV3ResNet34(DeepLabV3ResNet):
    def __init__(self, num_channels, num_classes):
        super(DeepLabV3ResNet34, self).__init__(34, num_channels, num_classes)


class DeepLabV3ResNet50(DeepLabV3ResNet):
    def __init__(self, num_channels, num_classes):
        super(DeepLabV3ResNet50, self).__init__(50, num_channels, num_classes)


class DeepLabV3ResNet101(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(DeepLabV3ResNet101, self).__init__(101, num_channels, num_classes)
