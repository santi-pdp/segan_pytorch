import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import *
from segan.datasets import *
import numpy as np
import random
import json
import os


def main(opts):
    segan = SEGAN(opts)     
    if opts.cuda:
        segan.cuda()
    if opts.mode == 'train':
        # create dataset and dataloader
        dset = SEDataset(opts.clean_trainset, 
                         opts.noisy_trainset, 
                         opts.preemph,
                         cache_dir=None,
                         max_samples=opts.max_samples)
        dloader = DataLoader(dset, batch_size=opts.batch_size,
                             shuffle=True, num_workers=opts.num_workers,
                             pin_memory=opts.cuda)
        criterion = nn.MSELoss()
        segan.train(opts, dloader, criterion, opts.l1_weight, 
                   opts.l1_dec_step, opts.l1_dec_epoch, opts.save_freq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--epoch', type=int, default=86)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--d_iter', type=int, default=1,
                        help='Number of k iterations (Def: 1).')
    parser.add_argument('--save_freq', type=int, default=50,
                        help="Batch save freq (Def: 50).")
    parser.add_argument('--canvas_size', type=int, default=(2 ** 14))
    parser.add_argument('--l1_dec_step', type=float, default=1e-5,
                        help='L1 regularization decay factor by batch ' \
                             '(Def: 1e-5).')
    parser.add_argument('--d_label_smooth', type=float, default=0.25,
                        help='Smoothing factor for D binary labels (Def: 0.25)')
    parser.add_argument('--save_path', type=str, default="segan_ckpt",
                        help="Path to save models (Def: segan_ckpt).")
    parser.add_argument('--g_lr', type=float, default=0.0002, 
                        help='Generator learning rate (Def: 0.0002).')
    parser.add_argument('--d_lr', type=float, default=0.0002, 
                        help='Discriminator learning rate (Def: 0.0002).')
    parser.add_argument('--beta_1', type=float, default=0.5,
                        help='Adam beta 1 (Def: 0.5).')
    parser.add_argument('--preemph', type=float, default=0.95,
                        help='Wav preemphasis factor (Def: 0.95).')
    parser.add_argument('--synthesis_path', type=str, default='segan_samples',
                        help='Path to save output samples (Def: ' \
                             'segan_samples).')
    parser.add_argument('--clean_trainset', type=str,
                        default='data/clean_trainset')
    parser.add_argument('--noisy_trainset', type=str,
                        default='data/noisy_trainset')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max num of samples to train (Def: None).')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode train/test (Def: train).')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader number of workers (Def: 2).')
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
    parser.add_argument('--g_opt', type=str, default='Adam')
    parser.add_argument('--d_opt', type=str, default='Adam')
    parser.add_argument('--g_dropout', type=float, default=0)
    parser.add_argument('--g_bnorm', action='store_true', default=False)
    parser.add_argument('--z_all', action='store_true', default=False)
    parser.add_argument('--z_dim', type=int, default=1024)
    parser.add_argument('--kwidth', type=int, default=31)
    parser.add_argument('--SND', action='store_true', default=False)
    parser.add_argument('--d_dropout', type=float, default=0)
    parser.add_argument('--d_bnorm', action='store_true', default=False)
    parser.add_argument('--skip', action='store_true', default=False)
    parser.add_argument('--D_rnn_pool', action='store_true', default=False)
    parser.add_argument('--l1_dec_epoch', type=int, default=10)
    parser.add_argument('--l1_weight', type=float, default=100,
                        help='L1 regularization weight (Def. 100). ')

    opts = parser.parse_args()

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    if opts.mode == 'train':
        # save opts
        with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
            cfg_f.write(json.dumps(vars(opts), indent=2))
    elif opts.mode == 'test':
        if not os.path.exists(opts.synthesis_path):
            os.makedirs(opts.synthesis_path)
    else:
        raise ValueError('Unrecognized mode: ', opts.mode)

    print('Parsed arguments: ', json.dumps(vars(opts), indent=2))

    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    main(opts)
