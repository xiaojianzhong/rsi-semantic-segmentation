import numpy as np

from .base import Dataset


class PatchedDataset(Dataset):
    def __init__(self, dataset, size, stride):
        super(PatchedDataset, self).__init__()
        self.dataset = dataset

        self.samples = []

        _, h, w = dataset[0][0].shape
        ph, pw = size
        sy, sx = stride

        for i in range(len(dataset)):
            for x1 in np.arange(0, w - pw + 1, sx):
                for y1 in np.arange(0, h - ph + 1, sy):
                    self.samples.append({
                        'index': i,
                        'x1': x1,
                        'y1': y1,
                        'x2': x1 + pw,
                        'y2': y1 + ph,
                    })

        self.image = None
        self.label = None
        self.index = -1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        index = sample['index']
        x1, y1, x2, y2 = sample['x1'], sample['y1'], sample['x2'], sample['y2']

        # cache
        if self.index != index:
            self.index = index
            self.image, self.label = self.dataset[index]

        image = self.image[:,y1:y2,x1:x2]
        label = self.label[y1:y2,x1:x2]

        return image, label

    @property
    def labels(self):
        return self.dataset.labels

    @property
    def pixels(self):
        return self.dataset.pixels

    @property
    def names(self):
        return self.dataset.names
