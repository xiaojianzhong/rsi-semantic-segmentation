import argparse
import logging
import os

import numpy as np
import torch
from skimage import io

from configs import CFG
from datas import build_dataset, build_transform
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('input',
                        type=str,
                        help='input image file')
    parser.add_argument('--output',
                        type=str,
                        default='output.tif',
                        help='output segmentation map file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device for inferring')
    parser.add_argument('--no-show',
                        action='store_true',
                        help='whether not to show output segmentation map')
    parser.add_argument('--no-save',
                        action='store_true',
                        help='whether not to save output segmentation map')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()

    # merge config with config file
    CFG.merge_from_file(args.config)

    # build transform
    transform = build_transform('test')
    # build dataset
    test_dataset = build_dataset('test')
    NUM_CHANNELS = test_dataset.num_channels
    NUM_CLASSES = test_dataset.num_classes
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(args.device)

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model']['state_dict'])
    best_miou = checkpoint['metric']['mIoU']
    logging.info('load checkpoint {} with mIoU={:.4f}'.format(args.checkpoint, best_miou))

    # infer
    model.eval()  # set model to evaluation mode

    x = io.imread(args.input)  # read image
    x = transform(image=x)['image']  # preprocess image
    x = x.unsqueeze(0)  # sample to batch
    x = x.to(args.device)

    y = model(x)

    pred = y.argmax(axis=1)
    pred = pred.data.cpu().numpy()
    pred = pred.squeeze(axis=0)
    pred = pred.astype(np.uint8)
    for label in test_dataset.labels:
        pred[pred == label] = test_dataset.label2pixel(label)

    if not args.no_show:
        io.imshow(pred)
        io.show()

    if not args.no_save:
        io.imsave(args.output, pred)


if __name__ == '__main__':
    main()
