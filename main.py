#!/usr/bin/env python3
import os
import os.path
import argparse
import torch
from data import DATASET_CONFIGS, TRAIN_DATASETS
from model import WGAN
from train import train
import utils


parser = argparse.ArgumentParser('PyTorch Implementation of WGAN-GP')
parser.add_argument(
    '--dataset', type=str,
    choices=list(TRAIN_DATASETS.keys()), default='cifar100'
)

parser.add_argument('--z-size', type=int, default=100)
parser.add_argument('--g-channel-size', type=int, default=64)
parser.add_argument('--c-channel-size', type=int, default=64)
parser.add_argument('--lamda', type=float, default=10.)

parser.add_argument('--lr', type=float, default=3e-05)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--sample-size', type=int, default=36)
parser.add_argument('--d-trains-per-g-train', type=int, default=5)

parser.add_argument('--sample-dir', type=str, default='samples')
parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
parser.add_argument('--loss-log-interval', type=int, default=30)
parser.add_argument('--image-log-interval', type=int, default=100)
parser.add_argument('--checkpoint-interval', type=int, default=1000)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

command = parser.add_mutually_exclusive_group(required=True)
command.add_argument('--test', action='store_true', dest='test')
command.add_argument('--train', action='store_false', dest='test')


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = torch.cuda.is_available() and args.cuda
    dataset = TRAIN_DATASETS[args.dataset]()
    dataset_config = DATASET_CONFIGS[args.dataset]

    wgan = WGAN(
        label=args.dataset,
        z_size=args.z_size,
        image_size=dataset_config['size'],
        image_channel_size=dataset_config['channels'],
        c_channel_size=args.c_channel_size,
        g_channel_size=args.g_channel_size,
    )

    utils.xavier_initialize(wgan)

    if cuda:
        wgan.cuda()

    if args.test:
        path = os.path.join(args.sample_dir, '{}-sample'.format(wgan.name))
        utils.load_checkpoint(wgan, args.checkpoint_dir)
        utils.test_model(wgan, args.sample_size, path)
    else:
        train(
            wgan, dataset,
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            lamda=args.lamda,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            epochs=args.epochs,
            d_trains_per_g_train=args.d_trains_per_g_train,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            image_log_interval=args.image_log_interval,
            loss_log_interval=args.loss_log_interval,
            resume=args.resume,
            cuda=cuda,
        )
