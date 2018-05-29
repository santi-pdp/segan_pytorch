import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import SEGAN
from segan.datasets import SEDataset, collate_fn
import numpy as np
import random
import json
import os


def main(opts):
    segan = SEGAN(opts)     
    if opts.g_pretrained_ckpt is not None:
        segan.G.load_pretrained(opts.g_pretrained_ckpt, True)
    if opts.d_pretrained_ckpt is not None:
        segan.D.load_pretrained(opts.d_pretrained_ckpt, True)
    if opts.cuda:
        segan.cuda()
    # create dataset and dataloader
    dset = SEDataset(opts.clean_trainset, 
                     opts.noisy_trainset, 
                     opts.preemph,
                     do_cache=True,
                     cache_dir=opts.cache_dir,
                     split='train',
                     stride=opts.data_stride,
                     max_samples=opts.max_samples,
                     verbose=True,
                     slice_workers=opts.slice_workers)
    # validation dataset 
    va_dset = SEDataset(opts.clean_valset, 
                        opts.noisy_valset, 
                        opts.preemph,
                        do_cache=True,
                        cache_dir=opts.cache_dir,
                        split='valid',
                        stride=opts.data_stride,
                        max_samples=opts.max_samples,
                        verbose=True,
                        slice_workers=opts.slice_workers)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, num_workers=opts.num_workers,
                         pin_memory=opts.cuda, collate_fn=collate_fn)
    va_dloader = DataLoader(va_dset, batch_size=opts.batch_size,
                            shuffle=False, num_workers=opts.num_workers,
                            pin_memory=opts.cuda, collate_fn=collate_fn)
    criterion = nn.MSELoss()
    segan.train(opts, dloader, criterion, opts.l1_weight,
                opts.l1_dec_step, opts.l1_dec_epoch,
                opts.save_freq,
                va_dloader=va_dloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="seganv1_ckpt",
                        help="Path to save models (Def: seganv1_ckpt).")
    parser.add_argument('--d_pretrained_ckpt', type=str, default=None,
                        help='Path to ckpt file to pre-load in training '
                             '(Def: None).')
    parser.add_argument('--g_pretrained_ckpt', type=str, default=None,
                        help='Path to ckpt file to pre-load in training '
                             '(Def: None).')
    parser.add_argument('--cache_dir', type=str, default='data')
    parser.add_argument('--clean_trainset', type=str,
                        default='data/clean_trainset')
    parser.add_argument('--noisy_trainset', type=str,
                        default='data/noisy_trainset')
    parser.add_argument('--clean_valset', type=str,
                        default='data/clean_valset')
    parser.add_argument('--noisy_valset', type=str,
                        default='data/noisy_valset')
    parser.add_argument('--data_stride', type=float,
                        default=0.5, help='Stride in seconds for data read')
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--epoch', type=int, default=86)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=50,
                        help="Batch save freq (Def: 50).")
    parser.add_argument('--canvas_size', type=int, default=(2 ** 14))
    parser.add_argument('--opt', type=str, default='rmsprop')
    parser.add_argument('--l1_dec_epoch', type=int, default=100)
    parser.add_argument('--l1_weight', type=float, default=100,
                        help='L1 regularization weight (Def. 100). ')
    parser.add_argument('--l1_dec_step', type=float, default=1e-5,
                        help='L1 regularization decay factor by batch ' \
                             '(Def: 1e-5).')
    parser.add_argument('--g_lr', type=float, default=0.0002, 
                        help='Generator learning rate (Def: 0.00005).')
    parser.add_argument('--d_lr', type=float, default=0.0002, 
                        help='Discriminator learning rate (Def: 0.00005).')
    parser.add_argument('--preemph', type=float, default=0.95,
                        help='Wav preemphasis factor (Def: 0.95).')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max num of samples to train (Def: None).')
    parser.add_argument('--g_act', type=str, default='prelu')
    parser.add_argument('--d_act', type=str, default='prelu')
    parser.add_argument('--skip_merge', type=str, default='concat')
    parser.add_argument('--skip_type', type=str, default='constant',
                        help='Type of skip connection: \n' \
                        '1) alpha: learn a vector of channels to ' \
                        ' multiply elementwise. \n' \
                        '2) conv: learn conv kernels of size 11 to ' \
                        ' learn complex responses in the shuttle.\n' \
                        '3) constant: with alpha value, set values to' \
                        ' not learnable, just fixed.')
    parser.add_argument('--skip_init', type=str, default='one',
                        help='Way to init skip connections')
    parser.add_argument('--eval_workers', type=int, default=2)
    parser.add_argument('--slice_workers', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1,
                        help='DataLoader number of workers (Def: 1).')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--g_enc_fmaps', type=int, nargs='+',
                        default=[16, 32, 32, 64, 64, 128, 128, \
                                 256, 256, 512, 1024],
                        help='Number of G encoder feature maps, ' \
                             '(Def: [16, 32, 32, 64, 64, 128, 128,' \
                             '256, 256, 512, 1024]).')
    parser.add_argument('--d_enc_fmaps', type=int, nargs='+',
                        default=[16, 32, 32, 64, 64, 128, 128, \
                                256, 256, 512, 1024],
                        help='Number of D encoder feature maps, ' \
                             '(Def: [16, 32, 32, 64, 64, 128, 128,' \
                              '256, 256, 512, 1024]).')
    parser.add_argument('--z_dim', type=int, default=1024)
    parser.add_argument('--SND', action='store_true', default=False)
    parser.add_argument('--g_snorm', action='store_true', default=False)
    parser.add_argument('--kwidth', type=int, default=31)
    parser.add_argument('--d_noise_epoch', type=int, default=3)
    parser.add_argument('--D_pool_size', type=int, default=8,
                        help='Dimension of last conv D layer time axis'
                             'prior to classifier real/fake (Def: 8)')
    parser.add_argument('--pooling_size', type=int, default=2,
                        help='Pool of every downsample/upsample '
                             'block in G or D (Def: 2).')
    parser.add_argument('--no_dbnorm', action='store_true', default=False)
    parser.add_argument('--alpha_val', type=float, default=1,
                        help='Alpha value for exponential avg of '
                             'validation curves (Def: 1)')

    opts = parser.parse_args()
    opts.d_bnorm = not opts.no_dbnorm

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    # save opts
    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
        cfg_f.write(json.dumps(vars(opts), indent=2))

    print('Parsed arguments: ', json.dumps(vars(opts), indent=2))

    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    main(opts)
