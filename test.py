import argparse
import logging
import os

import torch
from tqdm import tqdm

from configs import CFG
from datas import build_dataset, build_dataloader
from metric import Metric
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device for test')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()

    # merge config with config file
    CFG.merge_from_file(args.config)

    # log to stdout only
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ])

    # build dataset
    test_dataset = build_dataset('test')
    NUM_CHANNELS = test_dataset.num_channels
    NUM_CLASSES = test_dataset.num_classes
    # build data loader
    test_dataloader = build_dataloader(test_dataset, 'test')
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(args.device)
    # build metric
    metric = Metric(NUM_CLASSES)

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model']['state_dict'])
    best_miou = checkpoint['metric']['mIoU']
    logging.info('load checkpoint {} with mIoU={:.4f}'.format(args.checkpoint, best_miou))

    # test
    model.eval()  # set model to evaluation mode
    metric.reset()  # reset metric
    test_bar = tqdm(test_dataloader, desc='testing', ascii=True)
    with torch.no_grad():  # disable gradient back-propagation
        for x, label in test_bar:
            x, label = x.to(args.device), label.to(args.device)
            y = model(x)

            pred = y.argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())

            test_bar.set_postfix({
                'PA': f'{metric.PA():.4f}',
                'mPA': f'{metric.mPA():.4f}',
                'mIoU': f'{metric.mIoU():.4f}',
                'P': ','.join([f'{p:.4f}' for p in metric.Ps()]),
                'R': ','.join([f'{r:.4f}' for r in metric.Rs()]),
                'IoU': ','.join([f'{iou:.4f}' for iou in metric.IoUs()]),
            })


if __name__ == '__main__':
    main()
