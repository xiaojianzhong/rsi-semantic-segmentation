import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms


class Compose(nn.Module):
    def __init__(self, transforms):
        super(Compose, self).__init__()
        self.transforms = transforms

    def forward(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class RandomCrop(nn.Module):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, image, label):
        h, w, _ = image.shape
        new_h, new_w = self.size

        top = np.random.randint(0, h-new_h)
        down = top + new_h
        left = np.random.randint(0, w-new_w)
        right = left + new_w

        if image is not None:
            image = image[top:down, left:right, :]
        if label is not None:
            label = label[top:down, left:right]

        return image, label


class ToTensor(nn.Module):
    def __init__(self):
        super(ToTensor, self).__init__()
        self.to_tensor = transforms.ToTensor()

    def forward(self, image, label):
        image = self.to_tensor(image)
        return image, label


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, image, label):
        image = self.normalize(image)
        return image, label
