import argparse
import logging
import os
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from configs import CFG
from criterions import build_criterion
from datas import build_dataset, build_dataloader
from metric import Metric
from models import build_model
from optimizers import build_optimizer
from schedulers import build_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('--checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device for training')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S')),
                        help='path for experiment output files')
    parser.add_argument('--no-validate',
                        action='store_true',
                        help='whether not to validate in the training process')
    args = parser.parse_args()
    return args


def main():
    # merge arguments to config
    args = parse_args()
    CFG.merge_from_file(args.config)

    # create experiment output path if not exists
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())

    # log to file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'train.log')),
            logging.StreamHandler(),
        ])

    # initialize TensorBoard summary writer
    writer = SummaryWriter(logdir=args.path)

    # build dataset
    train_dataset = build_dataset('train')
    val_dataset = build_dataset('val')
    assert train_dataset.num_classes == val_dataset.num_classes
    NUM_CLASSES = train_dataset.num_classes
    # build data loader
    train_dataloader = build_dataloader(train_dataset, 'train')
    val_dataloader = build_dataloader(val_dataset, 'val')
    # build model
    model = build_model(NUM_CLASSES)
    model.to(args.device)
    # build criterion
    criterion = build_criterion()
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer = build_optimizer(model)
    # build scheduler
    scheduler = build_scheduler(optimizer)

    start_epoch = 0
    best_miou = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model']['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])
        start_epoch = checkpoint['optimizer']['epoch']
        best_miou = checkpoint['metric']['mIoU']
        logging.info('load checkpoint {} with mIoU={:.4f}'.format(args.checkpoint, best_miou))

    # train - validation loop
    for epoch in range(start_epoch, CFG.EPOCHS):
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr-epoch', lr, epoch)

        # train
        model.train()  # set model to training mode
        metric.reset()  # reset metric
        train_bar = tqdm(train_dataloader, desc='training', ascii=True)
        train_loss = 0.
        for batch, (x, label) in enumerate(train_bar):
            iteration = epoch * len(train_dataloader) + batch

            x, label = x.to(args.device), label.to(args.device)
            y = model(x)

            loss = criterion(y, label)
            train_loss += loss.item()
            train_bar.set_description('train loss: {:.4f}'.format(loss.item()))
            writer.add_scalar('train/loss-iteration', loss.item(), iteration+1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if NUM_CLASSES > 2:
                pred = y.data.cpu().numpy().argmax(axis=1)
            else:
                pred = (y.data.cpu().numpy() > 0.5).squeeze(1)
            label = label.data.cpu().numpy()
            metric.add(pred, label)
        train_loss /= len(train_dataloader)
        writer.add_scalar('train/loss-epoch', train_loss, epoch+1)

        pa, pas, mpa, ious, miou = metric.PA(), metric.PAs(), metric.mPA(), metric.IoUs(), metric.mIoU()
        writer.add_scalar('train/PA-epoch', pa, epoch+1)
        writer.add_scalar('train/mPA-epoch', mpa, epoch+1)
        writer.add_scalar('train/mIoU-epoch', miou, epoch+1)

        logging.info('train epoch={} | loss={:.4f} PA={:.4f} mPA={:.4f} mIoU={:.4f}'.format(epoch+1, train_loss, pa, mpa, miou))
        for c in range(NUM_CLASSES):
            logging.info('train epoch={} | class=#{} PA={:.4f} IoU={:.4f}'.format(epoch+1, c, pas[c], ious[c]))

        # validate
        if args.no_validate:
            continue
        model.eval()  # set model to evaluation mode
        metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for batch, (x, label) in enumerate(val_bar):
                x, label = x.to(args.device), label.to(args.device)
                y = model(x)

                loss = criterion(y, label)
                val_loss += loss.item()
                val_bar.set_description('val loss: {:.4f}'.format(loss.item()))

                if NUM_CLASSES > 2:
                    pred = y.data.cpu().numpy().argmax(axis=1)
                else:
                    pred = (y.data.cpu().numpy() > 0.5).squeeze(1)
                label = label.data.cpu().numpy()
                metric.add(pred, label)
        val_loss /= len(val_dataloader)
        writer.add_scalar('val/loss-epoch', val_loss, epoch+1)

        pa, pas, mpa, ious, miou = metric.PA(), metric.PAs(), metric.mPA(), metric.IoUs(), metric.mIoU()
        writer.add_scalar('val/PA-epoch', pa, epoch+1)
        writer.add_scalar('val/mPA-epoch', mpa, epoch+1)
        writer.add_scalar('val/mIoU-epoch', miou, epoch+1)

        logging.info('val epoch={} | loss={:.4f} PA={:.4f} mPA={:.4f} mIoU={:.4f}'.format(epoch+1, val_loss, pa, mpa, miou))
        for c in range(NUM_CLASSES):
            logging.info('val epoch={} | class=#{} PA={:.4f} IoU={:.4f}'.format(epoch+1, c, pas[c], ious[c]))

        # adjust learning rate if specified
        if scheduler is not None:
            scheduler.step(val_loss)  # TODO: remove val_loss

        # save checkpoint
        checkpoint = {
            'model': {
                'state_dict': model.state_dict(),
            },
            'optimizer': {
                'state_dict': optimizer.state_dict(),
                'epoch': epoch,
            },
            'metric': {
                'PA': pa,
                'PAs': pas,
                'mPA': mpa,
                'IoUs': ious,
                'mIoU': miou,
            },
        }
        torch.save(checkpoint, os.path.join(args.path, 'last.pth'))
        if miou > best_miou:
            torch.save(checkpoint, os.path.join(args.path, 'best.pth'))


if __name__ == '__main__':
    main()
