import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import SEGAN, SEGANDE, WSEGAN
from segan.datasets import SEDataset, collate_fn
from segan.utils import Additive
import numpy as np
import random
import json
import os


def main(opts):
    if opts.segande:
        segan = SEGANDE(opts)
    elif opts.wsegan:
        segan = WSEGAN(opts)
    else:
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
                     slice_size=opts.slice_size,
                     max_samples=opts.max_samples,
                     verbose=True,
                     slice_workers=opts.slice_workers,
                     preemph_norm=opts.preemph_norm,
                     random_scale=opts.random_scale
                    )
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, num_workers=opts.num_workers,
                         pin_memory=opts.cuda, 
                         collate_fn=collate_fn)
    if opts.clean_valset is not None:
        # validation dataset 
        va_dset = SEDataset(opts.clean_valset, 
                            opts.noisy_valset, 
                            opts.preemph,
                            do_cache=True,
                            cache_dir=opts.cache_dir,
                            split='valid',
                            stride=opts.data_stride,
                            slice_size=opts.slice_size,
                            max_samples=opts.max_samples,
                            verbose=True,
                            slice_workers=opts.slice_workers,
                            preemph_norm=opts.preemph_norm)
        va_dloader = DataLoader(va_dset, batch_size=300,
                                shuffle=False, num_workers=opts.num_workers,
                                pin_memory=opts.cuda, 
                                collate_fn=collate_fn)
    else:
        va_dloader = None
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
                        default=None)#'data/clean_valset')
    parser.add_argument('--noisy_valset', type=str,
                        default=None)#'data/noisy_valset')
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
    parser.add_argument('--d_pool_type', type=str, default='conv',
                        help='conv/rnn/none/gmax/gavg')
    parser.add_argument('--skip_init', type=str, default='one',
                        help='Way to init skip connections')
    parser.add_argument('--eval_workers', type=int, default=2)
    parser.add_argument('--slice_workers', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1,
                        help='DataLoader number of workers (Def: 1).')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--g_dec_fmaps', type=int, nargs='+',
                        default=None)
    parser.add_argument('--up_poolings', type=int, nargs='+',
                        default=None)
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
    parser.add_argument('--linterp', action='store_true', default=False)
    parser.add_argument('--SND', action='store_true', default=False)
    parser.add_argument('--g_snorm', action='store_true', default=False)
    parser.add_argument('--kwidth', type=int, default=31)
    parser.add_argument('--d_noise_epoch', type=int, default=3)
    parser.add_argument('--D_pool_size', type=int, default=8,
                        help='Dimension of last conv D layer time axis'
                             'prior to classifier real/fake (Def: 8)')
    parser.add_argument('--dkwidth', type=int, default=None,
                        help='Disc kwidth')
    parser.add_argument('--deckwidth', type=int, default=None,
                        help='G decoder kwidth')
    parser.add_argument('--dpooling_size', type=int, nargs='+', default=[2])
    parser.add_argument('--pooling_size', type=int, default=[2],
                        nargs='+',
                        help='Pool of every downsample/upsample '
                             'block in G or D (Def: 2).')
    parser.add_argument('--no_dbnorm', action='store_true', default=False)
    parser.add_argument('--convblock', action='store_true', default=False)
    parser.add_argument('--post_skip', action='store_true', default=False)
    parser.add_argument('--z_dropout', action='store_true', default=False)
    parser.add_argument('--pos_code', action='store_true', default=False,
                        help='Use positioning code in G')
    parser.add_argument('--alpha_val', type=float, default=1,
                        help='Alpha value for exponential avg of '
                             'validation curves (Def: 1)')
    parser.add_argument('--no_train_gen', action='store_true', default=False, 
                       help='Do NOT generate wav samples during training')
    parser.add_argument('--preemph_norm', action='store_true', default=False,
                        help='Inverts old  norm + preemph order in data ' \
                        'loading, so denorm has to respect this aswell')
    parser.add_argument('--segande', action='store_true', default=False,
                        help='Use Discriminator Enhanced')
    parser.add_argument('--wsegan', action='store_true', default=False)
    parser.add_argument('--vanilla_gan', action='store_true', default=False)
    parser.add_argument('--canvas_l2', type=float, default=0)
    parser.add_argument('--g_lnorm', action='store_true', default=False)
    parser.add_argument('--no_z', action='store_true', default=False)
    parser.add_argument('--no_skip', action='store_true', default=False)
    parser.add_argument('--satt', action='store_true', default=False)
    parser.add_argument('--mlpconv', action='store_true', default=False)
    parser.add_argument('--slice_size', type=int, default=16384)
    parser.add_argument('--random_scale', type=float, nargs='+', 
                        default=[1], help='Apply randomly a scaling factor' \
                                          'in list to the (clean, noisy) pair')
    parser.add_argument('--pow_weight', type=float, default=0.001)
    parser.add_argument('--phase_shift', type=int, default=5)
    parser.add_argument('--misalign_pair', action='store_true', default=False)
    parser.add_argument('--comb_net', action='store_true', default=False)
    parser.add_argument('--out_gate', action='store_true', default=False)
    parser.add_argument('--big_out_filter', action='store_true', default=False)
    parser.add_argument('--hidden_comb', action='store_true', default=False)
    parser.add_argument('--nigenerator', action='store_true', default=False)
    parser.add_argument('--noises_dir', type=str,
                        default='data/silent/additive_noises')
    parser.add_argument('--linterp_mode', type=str, default='linear')
    parser.add_argument('--no_bias', action='store_true', default=False,
                        help='Disable all biases in Generator')
    parser.add_argument('--z_std', type=float, default=1,
                        help='Apply std multiplication to z Normal prior')
    parser.add_argument('--ardiscriminator', action='store_true',
                        default=False)
    parser.add_argument('--n_fft', type=int, default=2048)

    opts = parser.parse_args()
    opts.d_bnorm = not opts.no_dbnorm
    opts.bias = not opts.no_bias

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
