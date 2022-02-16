import albumentations as A
import albumentations.pytorch
import torch.distributed as dist
from torch.utils.data import DataLoader

from configs import CFG
from .gf2_building import GF2BuildingDataset
from .massachusetts_building import MassachusettsBuildingDataset
from .transform import TransformDataset


def build_transform(split):
    transforms = []
    if split == 'train':
        pass
    elif split == 'val' or split == 'test':
        # TODO: report bug
        transforms.append(A.PadIfNeeded(min_height=None,
                                        min_width=None,
                                        pad_height_divisor=32,
                                        pad_width_divisor=32))
    transforms.append(A.Normalize(mean=CFG.DATASET.MEANS, std=CFG.DATASET.STDS))
    transforms.append(A.pytorch.ToTensorV2())
    transform = A.Compose(transforms)
    return transform


def build_dataset(split):
    assert split in ['train', 'val', 'test']
    if CFG.DATASET.NAME == 'gf2-building':
        dataset = GF2BuildingDataset(CFG.DATASET.ROOT, split)
    elif CFG.DATASET.NAME == 'massachusetts-building':
        dataset = MassachusettsBuildingDataset(CFG.DATASET.ROOT, split)
    else:
        raise NotImplementedError('invalid dataset: {}'.format(CFG.DATASET.NAME))
    dataset = TransformDataset(dataset, transform=build_transform(split))
    return dataset


def build_dataloader(dataset, sampler, split):
    assert split in ['train', 'val', 'test']
    if split == 'train':
        return DataLoader(dataset,
                          sampler=sampler,
                          batch_size=CFG.DATALOADER.BATCH_SIZE // dist.get_world_size(),
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True)
    elif split == 'val':
        return DataLoader(dataset,
                          sampler=sampler,
                          batch_size=1,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True)
    elif split == 'test':
        return DataLoader(dataset,
                          sampler=sampler,
                          batch_size=1,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True)
