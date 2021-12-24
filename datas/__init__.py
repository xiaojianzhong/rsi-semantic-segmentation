from torch.utils.data import DataLoader

import datas.transforms as transforms
from configs import CFG
from .gf2_building import GF2BuildingDataset
from .massachusetts_building import MassachusettsBuildingDataset
from .patch import PatchedDataset


def build_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG.DATASET.MEANS, std=CFG.DATASET.STDS),
    ])
    return transform


def build_dataset(split):
    assert split in ['train', 'val', 'test']
    if CFG.DATASET.NAME == 'gf2-building':
        dataset = GF2BuildingDataset(CFG.DATASET.ROOT,
                                     split,
                                     transform=build_transform())
    elif CFG.DATASET.NAME == 'massachusetts-building':
        dataset = MassachusettsBuildingDataset(CFG.DATASET.ROOT,
                                               split,
                                               transform=build_transform())
    else:
        raise NotImplementedError('invalid dataset: {}'.format(CFG.DATASET.NAME))
    dataset = PatchedDataset(dataset,
                             (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                             (CFG.DATASET.PATCH.STRIDE_Y, CFG.DATASET.PATCH.STRIDE_X))
    return dataset


def build_dataloader(dataset, split):
    assert split in ['train', 'val', 'test']
    if split == 'train':
        return DataLoader(dataset,
                          batch_size=CFG.DATALOADER.BATCH_SIZE,
                          shuffle=True,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True)
    elif split == 'val':
        return DataLoader(dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True)
    elif split == 'test':
        return DataLoader(dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True)
