import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from apex import amp
from apex.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
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
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S')),
                        help='path for experiment output files')
    parser.add_argument('--no-validate',
                        action='store_true',
                        help='whether not to validate in the training process')
    parser.add_argument('-n',
                        '--nodes',
                        type=int,
                        default=1,
                        help='number of nodes / machines')
    parser.add_argument('-g',
                        '--gpus',
                        type=int,
                        default=1,
                        help='number of GPUs per node / machine')
    parser.add_argument('-r',
                        '--rank-node',
                        type=int,
                        default=0,
                        help='ranking of the current node / machine')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='backend for PyTorch DDP')
    parser.add_argument('--master-ip',
                        type=str,
                        default='localhost',
                        help='network IP of the master node / machine')
    parser.add_argument('--master-port',
                        type=str,
                        default='8888',
                        help='network port of the master process on the master node / machine')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed')
    parser.add_argument('--opt-level',
                        type=str,
                        default='O0',
                        help='optimization level for nvidia/apex')
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus
    return args


def worker(rank_gpu, args):
    rank_process = args.gpus * args.rank_node + rank_gpu
    # initialize process group
    dist.init_process_group(backend=args.backend,
                            init_method=f'tcp://{args.master_ip}:{args.master_port}',
                            world_size=args.world_size,
                            rank=rank_process)
    logging.info('train on {} processes'.format(dist.get_world_size()))

    # use device cuda:n in the process #n
    torch.cuda.set_device(rank_gpu)
    device = torch.device('cuda', rank_gpu)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # initialize TensorBoard summary writer on the master process
    if dist.get_rank() == 0:
        writer = SummaryWriter(logdir=args.path)

    # build dataset
    train_dataset = build_dataset('train')
    val_dataset = build_dataset('val')
    assert train_dataset.num_classes == val_dataset.num_classes
    NUM_CHANNELS = train_dataset.num_channels
    NUM_CLASSES = train_dataset.num_classes
    # build data sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # build data loader
    train_dataloader = build_dataloader(train_dataset, train_sampler, 'train')
    val_dataloader = build_dataloader(val_dataset, None, 'val')
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    # build criterion
    criterion = build_criterion()
    criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer = build_optimizer(model)
    # build scheduler
    scheduler = build_scheduler(optimizer)

    # mixed precision
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    # DDP
    model = DistributedDataParallel(model)

    epoch = 0
    iteration = 0
    best_miou = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.module.load_state_dict(checkpoint['model']['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])
        epoch = checkpoint['optimizer']['epoch']
        iteration = checkpoint['optimizer']['iteration']
        best_miou = checkpoint['metric']['mIoU']
        logging.info('load checkpoint {} with mIoU={:.4f}'.format(args.checkpoint, best_miou))

    # train - validation loop
    while True:
        epoch += 1
        if epoch > CFG.EPOCHS:
            if dist.get_rank() == 0:
                writer.close()
            return

        train_dataloader.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train
        model.train()  # set model to training mode
        metric.reset()  # reset metric
        train_bar = tqdm(train_dataloader, desc='training', ascii=True)
        train_loss = 0.
        for x, label in train_bar:
            iteration += 1

            x, label = x.to(device), label.to(device)
            y = model(x)

            loss = criterion(y, label)
            train_loss += loss.item()
            if dist.get_rank() == 0:
                writer.add_scalar('train/loss-iteration', loss.item(), iteration)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            pred = y.argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())

            train_bar.set_postfix({
                'epoch': epoch,
                'loss': f'{loss.item():.4f}',
                'PA': f'{metric.PA():.4f}',
                'mPA': f'{metric.mPA():.4f}',
                'mIoU': f'{metric.mIoU():.4f}',
                'P': ','.join([f'{p:.4f}' for p in metric.Ps()]),
                'R': ','.join([f'{r:.4f}' for r in metric.Rs()]),
                'IoU': ','.join([f'{iou:.4f}' for iou in metric.IoUs()]),
            })
        train_loss /= len(train_dataloader)
        if dist.get_rank() == 0:
            writer.add_scalar('train/loss-epoch', train_loss, epoch)

        pa, mpa, miou, ps, rs, ious = metric.PA(), metric.mPA(), metric.mIoU(), metric.Ps(), metric.Rs(), metric.IoUs()
        if dist.get_rank() == 0:
            writer.add_scalar('train/PA-epoch', pa, epoch)
            writer.add_scalar('train/mPA-epoch', mpa, epoch)
            writer.add_scalar('train/mIoU-epoch', miou, epoch)

        # validate
        if args.no_validate:
            continue
        model.eval()  # set model to evaluation mode
        metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for x, label in val_bar:
                x, label = x.to(device), label.to(device)
                y = model(x)

                loss = criterion(y, label)
                val_loss += loss.item()

                pred = y.argmax(axis=1)
                metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())

                val_bar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{loss.item():.4f}',
                    'PA': f'{metric.PA():.4f}',
                    'mPA': f'{metric.mPA():.4f}',
                    'mIoU': f'{metric.mIoU():.4f}',
                    'P': ','.join([f'{p:.4f}' for p in metric.Ps()]),
                    'R': ','.join([f'{r:.4f}' for r in metric.Rs()]),
                    'IoU': ','.join([f'{iou:.4f}' for iou in metric.IoUs()]),
                })
        val_loss /= len(val_dataloader)
        if dist.get_rank() == 0:
            writer.add_scalar('val/loss-epoch', val_loss, epoch)

        pa, mpa, miou, ps, rs, ious = metric.PA(), metric.mPA(), metric.mIoU(), metric.Ps(), metric.Rs(), metric.IoUs()
        if dist.get_rank() == 0:
            writer.add_scalar('val/PA-epoch', pa, epoch)
            writer.add_scalar('val/mPA-epoch', mpa, epoch)
            writer.add_scalar('val/mIoU-epoch', miou, epoch)

        # adjust learning rate if specified
        if scheduler is not None:
            scheduler.step(val_loss)  # TODO: remove val_loss

        # save checkpoint on the master process
        if dist.get_rank() == 0:
            checkpoint = {
                'model': {
                    'state_dict': model.state_dict(),
                },
                'optimizer': {
                    'state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'iteration': iteration,
                },
                'metric': {
                    'PA': pa,
                    'mPA': mpa,
                    'mIoU': miou,
                    'Ps': ps,
                    'Rs': rs,
                    'IoUs': ious,
                },
            }
            torch.save(checkpoint, os.path.join(args.path, 'last.pth'))
            if miou > best_miou:
                best_miou = miou
                torch.save(checkpoint, os.path.join(args.path, 'best.pth'))


def main():
    # parse command line arguments
    args = parse_args()

    # create experiment output path if not exists
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())

    # log to stdout only
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
        ])

    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
