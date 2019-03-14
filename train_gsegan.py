import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from segan.models import GSEGAN
from segan.models import SEGAN, WSEGAN, GSEGAN
from segan.datasets import SEOnlineDataset
from segan.datasets import collate_fn
from segan.transforms import *
from waveminionet.models.frontend import wf_builder
from torchvision.transforms import Compose
import soundfile as sf
import numpy as np
import random
import json
import os


def main(opts):
    # select device to work on 
    device = 'cpu'
    if torch.cuda.is_available and not opts.no_cuda:
        device = 'cuda'
        opts.cuda = True
    CUDA = (device == 'cuda')
    num_devices = 1
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if CUDA:
        torch.cuda.manual_seed_all(opts.seed)
        num_devices = torch.cuda.device_count()
        opts.num_devices = num_devices
        print('[*] Using CUDA {} devices'.format(num_devices))
    else:
        print('[!] Using CPU')
    print('Seeds initialized to {}'.format(opts.seed))

    # frontend will be None by default
    frontend = None
    if opts.wsegan:
        segan = WSEGAN(opts)
        if opts.wseganfe_cfg is not None:
            print('+' * 30)
            print('Building WSEGAN frontend {}'.format(opts.wseganfe_cfg))
            frontend = wf_builder(opts.wseganfe_cfg)
            assert opts.wseganfe_ckpt is not None
            frontend.load_pretrained(opts.wseganfe_ckpt, load_last=True,
                                     verbose=True)
            frontend.eval()
            frontend.to(device)
            print('+' * 30)
    else:
        segan = GSEGAN(opts)
    #segan.to(device)
    print(segan)
    # possibly load pre-trained sections of networks G or D
    print('Total model parameters: ',  segan.get_n_params())
    if opts.g_pretrained_ckpt is not None:
        segan.G.load_pretrained(opts.g_pretrained_ckpt, True)
    if opts.d_pretrained_ckpt is not None:
        segan.D.load_pretrained(opts.d_pretrained_ckpt, True)

    # Build chunker transform with proper slice size
    chunker = SingleChunkWav(opts.slice_size)

    # Build transforms
    trans = PCompose([
        Resample(opts.resample_factors),
        Clipping(),
        Chopper(max_chops=5),
        #Additive('data/noises/train')
    ])

    # create Dataset(s) and Dataloader(s)
    dset = SEOnlineDataset(opts.data_root,
                           distorteds=opts.distorted_roots,
                           chunker=chunker,
                           nsamples=opts.data_samples,
                           transform=trans,
                           utt2class=opts.utt2class)
    dloader = DataLoader(dset, batch_size=opts.batch_size,
                         shuffle=True, num_workers=opts.num_workers,
                         #collate_fn=collate_fn,
                         pin_memory=CUDA)
    va_dloader = None

    nsamples = dset.total_samples
    criterion = nn.MSELoss()
    segan.train(opts, dloader, criterion, opts.l1_weight,
                opts.l1_dec_step, opts.l1_dec_epoch,
                opts.save_freq,
                tr_samples=nsamples,
                va_dloader=va_dloader, frontend=frontend,
                device=device)


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
    parser.add_argument('--data_root', type=str,
                        default='data_gsegan/clean_trainset_trimsil')
    parser.add_argument('--data_samples', type=int, default=1186688000,
                        help='Number of wav samples in the data root. '
                             'Computed externally with soxi for efficiency '
                             'for the default directory. If zero, it will '
                             'be recalculated reading wavs.')
    parser.add_argument('--distorted_roots', type=str, nargs='+',
                        default=None)
    parser.add_argument('--resample_factors', type=int, nargs='+',
                        default=[2, 4, 8])
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=50,
                        help="Batch save freq (Def: 50).")
    parser.add_argument('--slice_size', type=int, default=16384)
    parser.add_argument('--opt', type=str, default='rmsprop')
    parser.add_argument('--l1_dec_epoch', type=int, default=100)
    parser.add_argument('--l1_weight', type=float, default=100,
                        help='L1 regularization weight (Def. 100). ')
    parser.add_argument('--l1_dec_step', type=float, default=1e-5,
                        help='L1 regularization decay factor by batch ' \
                             '(Def: 1e-5).')
    parser.add_argument('--g_lr', type=float, default=0.00005, 
                        help='Generator learning rate (Def: 0.00005).')
    parser.add_argument('--d_lr', type=float, default=0.00005, 
                        help='Discriminator learning rate (Def: 0.0005).')
    parser.add_argument('--preemph', type=float, default=0.95,
                        help='Wav preemphasis factor (Def: 0.95).')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max num of samples to train (Def: None).')
    parser.add_argument('--eval_workers', type=int, default=2)
    parser.add_argument('--slice_workers', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1,
                        help='DataLoader number of workers (Def: 1).')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA even if device is available')
    parser.add_argument('--random_scale', type=float, nargs='+', 
                        default=[1], help='Apply randomly a scaling factor' \
                                          'in list to the (clean, noisy) pair')
    parser.add_argument('--no_train_gen', action='store_true', default=False, 
                       help='Do NOT generate wav samples during training')
    parser.add_argument('--preemph_norm', action='store_true', default=False,
                        help='Inverts old  norm + preemph order in data ' \
                        'loading, so denorm has to respect this aswell')
    parser.add_argument('--no_bias', action='store_true', default=False,
                        help='Disable all biases in Generator')
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--reg_loss', type=str, default='l1_loss',
                        help='Regression loss (l1_loss or mse_loss) in the '
                             'output of G (Def: l1_loss)')
    parser.add_argument('--critic_iters', type=int, default=5)
    parser.add_argument('--gp_weight', type=float, default=10)

    # Skip connections options for G
    parser.add_argument('--skip_merge', type=str, default='concat')
    parser.add_argument('--skip_type', type=str, default='alpha',
                        help='Type of skip connection: \n' \
                        '1) alpha: learn a vector of channels to ' \
                        ' multiply elementwise. \n' \
                        '2) conv: learn conv kernels of size 11 to ' \
                        ' learn complex responses in the shuttle.\n' \
                        '3) constant: with alpha value, set values to' \
                        ' not learnable, just fixed.\n(Def: alpha)')
    parser.add_argument('--skip_init', type=str, default='one',
                        help='Way to init skip connections (Def: one)')
    parser.add_argument('--skip_kwidth', type=int, default=31)

    # Generator parameters
    parser.add_argument('--gkwidth', type=int, default=31)
    parser.add_argument('--genc_fmaps', type=int, nargs='+',
                        default=[64, 128, 256, 512, 1024],
                        help='Number of G encoder feature maps, ' \
                             '(Def: [64, 128, 256, 512, 1024]).')
    parser.add_argument('--genc_poolings', type=int, nargs='+',
                        default=[4, 4, 4, 4, 4],
                        help='G encoder poolings')
    parser.add_argument('--z_dim', type=int, default=1024)
    parser.add_argument('--gdec_fmaps', type=int, nargs='+',
                        default=[512, 256, 128])
    parser.add_argument('--gdec_poolings', type=int, nargs='+',
                        default=[4, 4, 10])
    parser.add_argument('--gdec_kwidth', type=int, nargs='+',
                        default=[11, 11, 11, 11, 20])
    parser.add_argument('--gnorm_type', type=str, default=None,
                        help='Normalization to be used in G. Can '
                        'be: (1) snorm, (2) bnorm or (3) none '
                        '(Def: None).')
    parser.add_argument('--no_z', action='store_true', default=False)
    parser.add_argument('--no_skip', action='store_true', default=False)
    parser.add_argument('--vanilla_gan', action='store_true', default=False)
    parser.add_argument('--pow_weight', type=float, default=0.001)
    parser.add_argument('--fe_weight', type=float, default=0.001)
    parser.add_argument('--misalign_pair', action='store_true', default=False)
    parser.add_argument('--interf_pair', action='store_true', default=False)

    # Discriminator parameters
    parser.add_argument('--denc_fmaps', type=int, nargs='+',
                        default=[64, 128, 256, 512, 1024],
                        help='Number of D encoder feature maps, ' \
                             '(Def: [64, 128, 256, 512, 1024]')
    parser.add_argument('--dpool_type', type=str, default='mha',
                        help='mlp/mha (Def: mha)')
    parser.add_argument('--dpool_slen', type=int, default=16,
                        help='Dimension of last conv D layer time axis'
                             'prior to classifier real/fake (Def: 16)')
    parser.add_argument('--dkwidth', type=int, default=None,
                        help='Disc kwidth (Def: None), None is gkwidth.')
    parser.add_argument('--denc_poolings', type=int, nargs='+', 
                        default=[4, 4, 4, 4, 4],
                        help='(Def: [4, 4, 4, 4, 4])')
    parser.add_argument('--dnorm_type', type=str, default='bnorm',
                        help='Normalization to be used in D. Can '
                        'be: (1) snorm, (2) bnorm or (3) none '
                        '(Def: bnorm).')
    parser.add_argument('--nheads', type=int, default=4,
                        help='Number of attention heads (Def: 4).')
    parser.add_argument('--dffn_size', type=int, default=128,
                        help='Feed-forward network size after MHA (Def: 128).')
    parser.add_argument('--dW_size', type=int, default=128,
                        help='Embedding size prior to MHA (Def: 128).')
    parser.add_argument('--gfe_cfg', type=str, default=None,
                        help='Frontend config file for Generator (Def: None).')
    parser.add_argument('--dfe_cfg', type=str, default=None,
                        help='Frontend config file for Discriminator '
                        '(Def: None).')
    parser.add_argument('--gfe_ckpt', type=str, default=None,
                        help='Pretrained ckpt of G frontend (Def: None).')
    parser.add_argument('--no-gfeft', action='store_true', default=False,
                        help='Do not fine tune encoder in G (Def: False).')
    parser.add_argument('--no-dfeft', action='store_true', default=False,
                        help='Do not fine tune encoder in D (Def: False).')
    parser.add_argument('--dfe_ckpt', type=str, default=None,
                        help='Pretrained ckpt of D frontend (Def: None).')
    parser.add_argument('--only_dfe', action='store_true', default=False,
                        help='Only use D to update frontend for both G and D'
                             ' (Def: False).')

    parser.add_argument('--patience', type=int, default=100,
                        help='If validation path is set, there are '
                             'denoising evaluations running for which '
                             'COVL, CSIG, CBAK, PESQ and SSNR are '
                             'computed. Patience is number of validation '
                             'epochs to wait til breakining train loop. This '
                             'is an unstable and slow process though, so we'
                             'avoid patience by setting it high atm (Def: 100).'
                       )
    parser.add_argument('--phase_shift', type=int, default=5)
    parser.add_argument('--sinc_conv', action='store_true', default=False)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--wsegan', action='store_true', default=False)
    parser.add_argument('--res_deconv', action='store_true', default=False,
                        help='Apply residual deconv blocks (Def: False).')
    parser.add_argument('--gdec_type', type=str, default='deconv',
                        help='deconv/resdeconv (Def: deconv).')

    parser.add_argument('--wseganfe_cfg', type=str, default=None,
                        help='Frontend config file for WSEGAN (Def: None).')
    parser.add_argument('--wseganfe_ckpt', type=str, default=None,
                        help='Frontend ckpt file for WSEGAN (Def: None).')
    parser.add_argument('--step_iters', type=int, default=1,
                        help='Run optimizer update after this counter')
    parser.add_argument('--utt2class', type=str, default=None,
                        help='Dictionary mapping each utterance '
                             'basename to a class (Def: None).')

    opts = parser.parse_args()
    opts.bias = not opts.no_bias
    opts.gfe_ft = not opts.no_gfeft
    opts.dfe_ft = not opts.no_dfeft

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    opts.num_classes = None
    if opts.utt2class is not None:
        with open(opts.utt2class, 'r') as f:
            utt2class = json.load(f)
            opts.num_classes = max(utt2class.values()) + 1

    # save opts
    with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
        cfg_f.write(json.dumps(vars(opts), indent=2))

    print('Parsed arguments: ', json.dumps(vars(opts), indent=2))
    main(opts)
