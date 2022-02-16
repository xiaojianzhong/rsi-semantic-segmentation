import os

import numpy as np

from .base import Dataset


class GF2BuildingDataset(Dataset):
    def __init__(self, root, split):
        super(GF2BuildingDataset, self).__init__()
        assert split in ['train', 'val', 'test']

        self.image_paths = []
        self.label_paths = []
        for _, _, filenames in os.walk(root, os.path.join(root, 'data')):
            for filename in filenames:
                image_path = os.path.join(root, 'data', filename)
                self.image_paths.append(image_path)
                label_path = os.path.join(root, 'labels', filename)
                self.label_paths.append(label_path)
        assert len(self.image_paths) == len(self.label_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path, label_path = self.image_paths[i], self.label_paths[i]

        image = np.load(image_path).transpose((1, 2, 0))
        label = np.load(label_path).argmax(axis=0)

        return image, label

    @property
    def num_channels(self):
        return 4

    @property
    def labels(self):
        return [0, 1, 2]

    @property
    def pixels(self):
        return [0, 128, 255]

    @property
    def names(self):
        return ['background', 'building', 'factory']
