import os

import numpy as np
import pandas as pd
from skimage import io

from .base import Dataset


class MassachusettsBuildingDataset(Dataset):
    def __init__(self, root, split):
        super(MassachusettsBuildingDataset, self).__init__()
        assert split in ['train', 'val', 'test']

        metadata_path = os.path.join(root, 'metadata.csv')
        df = pd.read_csv(metadata_path).query('split == "{}"'.format(split))
        self.image_paths = [os.path.join(root, image_path) for image_path in df['tiff_image_path']]
        self.label_paths = [os.path.join(root, label_path) for label_path in df['tif_label_path']]
        assert len(self.image_paths) == len(self.label_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image_path, label_path = self.image_paths[i], self.label_paths[i]

        image, label = io.imread(image_path), io.imread(label_path, as_gray=True).astype(np.uint8)
        for pixel in self.pixels:
            label[label == pixel] = self.pixel2label(pixel)

        return image, label

    @property
    def num_channels(self):
        return 3

    @property
    def labels(self):
        return [0, 1]

    @property
    def pixels(self):
        return [0, 255]

    @property
    def names(self):
        return ['background', 'building']
