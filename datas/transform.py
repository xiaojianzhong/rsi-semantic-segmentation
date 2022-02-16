import torch.utils as utils


class TransformDataset(utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        super(TransformDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        return image, label

    @property
    def num_channels(self):
        return self.dataset.num_channels

    @property
    def labels(self):
        return self.dataset.labels

    @property
    def pixels(self):
        return self.dataset.pixels

    @property
    def names(self):
        return self.dataset.names
