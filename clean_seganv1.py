import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import *
from segan.datasets import *
import soundfile as sf
from scipy.io import wavfile
from torch.autograd import Variable
import numpy as np
import random
import librosa
import matplotlib
import timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import os


def make_divN(tensor, N):
    # make tensor time dim divisible by N (for good decimation)
    pad_num = (tensor.size(1) + N) - (tensor.size(1) % N) - tensor.size(1)
    pad = torch.zeros(tensor.size(0), pad_num, tensor.size(-1))
    print('tensor size: ', tensor.size())
    print('pad size: ', pad.size())
    return torch.cat((tensor, pad), dim=1)

def show_spec(signal, n_fft=512):
    D = librosa.stft(signal, n_fft=512)
    # fft results
    X = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(10, 10))
    plt.imshow(X)
    plt.title('Power spec')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

def batchize_wav(wav):
    chunks = []
    N = 16384
    M = N
    for beg in range(0, wav.shape[0], N):
        if wav.shape[0] - beg < N:
            break
        chunks.append(wav[beg:beg + N])
    return chunks

def main(opts):
    segan = SEGAN(opts)     
    if opts.pretrained_ckpt is not None:
        segan.load_pretrained(opts.pretrained_ckpt, True)
    #segan.load_raw_weights(opts.pretrained_ckpt)
    if opts.cuda:
        segan.cuda()
    assert opts.test_dir is not None
    segan.G.eval()
    # process every wav in the test_dir
    if len(opts.test_dir) == 1:
        # assume we read directory
        twavs = glob.glob(os.path.join(opts.test_dir[0], '*.wav'))
    else:
        twavs = opts.test_dir
    beg_t = timeit.default_timer()
    batch = []
    seqlens = []
    cbatch = []
    cseqlens = []
    maxlen = 0
    cmaxlen = 0
    for t_i, twav in enumerate(twavs, start=1):
        tbname = os.path.basename(twav)
        rate, wav = wavfile.read(twav)
        wav = normalize_wave_minmax(wav)
        wav = pre_emphasize(wav, opts.preemph)
        #wav = abs_normalize_wave_minmax(wav)
        #wav = wav * 0.1 + 0.2
        #owav = de_emphasize(wav, opts.preemph)
        #out_path = os.path.join(opts.synthesis_path,
        #                        'ori_' + tbname)
        #sf.write(out_path, owav, 16000, 'PCM_16')
        #wav, rate = librosa.load(twav, 16000)
        #wav = torch.FloatTensor(batchize_wav(wav)).view(-1, 1, 16384)
        #print('wav shape: ', wav)
        pwav = torch.FloatTensor(wav).view(1,1,-1)
        #seqlen = wav.size(1)
        #pwav = make_divN(wav, 2048)
        #pwav = wav[:, :16384, :].contiguous()
        #pwav = Variable(pwav).view(1,1,-1)
        if opts.cuda:
            pwav = pwav.cuda()
        g_wav, g_c = segan.generate(pwav)
        #g_wav_eval = g_wav.view(g_wav.size(0), -1)
        print('g_wav shape: ', g_wav.shape)
        #g_wav = g_wav[:seqlen].squeeze().cpu().data.numpy()
        #g_wav = g_wav - g_wav.mean()
        #g_wav = g_wav / np.max(np.abs(g_wav))
        #g_wav = g_wav.view(1,1,-1)
        out_path = os.path.join(opts.synthesis_path,
                                tbname) 
        #preemph_out_path = os.path.join(opts.synthesis_path,
        #                        'preemph_'+tbname) 
        #norm_out_path = os.path.join(opts.synthesis_path,
        #                        'norm_'+tbname) 
        #gc_out_path = os.path.join(opts.synthesis_path,
        #                        'gc_'+tbname) 
        print('g_wav min: ', g_wav.min())
        print('g_wav max: ', g_wav.max())
        print('g_wav std: ', g_wav.std())
        wavfile.write(out_path, 16000, g_wav)
        #if preemph_wav is not None:
        #    wavfile.write(preemph_out_path, 16000, preemph_wav)
        #if norm_wav is not None:
        #    wavfile.write(norm_out_path, 16000, norm_wav)
        #np.save(gc_out_path, g_c.data.numpy())
        #sf.write(out_path, g_wav,
        #         16000, 'PCM_16')
        #wavfile.write(out_path, 16000, 
        #              g_wav.squeeze().cpu().data.numpy())
        end_t = timeit.default_timer()
        print('Clenaed {}/{}: {} in {} s'.format(t_i, len(twavs), twav,
                                                 end_t-beg_t))
        beg_t = timeit.default_timer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None,
                        help='Filename of wav to test')
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--epoch', type=int, default=86)
    parser.add_argument('--segan_type', type=str, default='silent')
    parser.add_argument('--critic_iters', type=int, default=5,
                        help='For WSilentSEGAN, num of D updates.')
    parser.add_argument('--iters', type=int, default=20000,
                       help='For WSilentSEGAN, num of train iters.')
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
    parser.add_argument('--g_optim', type=str, default='rmsprop')
    parser.add_argument('--d_optim', type=str, default='rmsprop')
    parser.add_argument('--g_lr', type=float, default=0.0002, 
                        help='Generator learning rate (Def: 0.0002).')
    parser.add_argument('--d_lr', type=float, default=0.0002, 
                        help='Discriminator learning rate (Def: 0.0002).')
    parser.add_argument('--beta_1', type=float, default=0.5,
                        help='Adam beta 1 (Def: 0.5).')
    parser.add_argument('--preemph', type=float, default=0.95,
                        help='Wav preemphasis factor (Def: 0.95).')
    parser.add_argument('--test_dir', type=str, nargs='+', default=None)
    parser.add_argument('--test_clean_dir', type=str, default=None)
    parser.add_argument('--synthesis_path', type=str, default='segan_samples',
                        help='Path to save output samples (Def: ' \
                             'segan_samples).')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max num of samples to train (Def: None).')
    parser.add_argument('--g_act', type=str, default='prelu')
    parser.add_argument('--d_act', type=str, default='lrelu')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode train/test (Def: train).')
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
    parser.add_argument('--skip_blacklist', type=int, nargs='+',
                        default=[],
                        help='Remove skip connection in this list indices'
                             '(Def: []).')
    parser.add_argument('--g_opt', type=str, default='Adam')
    parser.add_argument('--d_opt', type=str, default='Adam')
    parser.add_argument('--wd', type=float, default=0.)
    parser.add_argument('--g_dropout', type=float, default=0)
    parser.add_argument('--core2d', action='store_true', default=False)
    parser.add_argument('--g_bnorm', action='store_true', default=False)
    parser.add_argument('--no_g_bias', action='store_true', default=False)
    parser.add_argument('--z_all', action='store_true', default=False)
    parser.add_argument('--z_dim', type=int, default=1024)
    parser.add_argument('--kwidth', type=int, default=31)
    parser.add_argument('--dec_kwidth', type=int, default=None)
    parser.add_argument('--core2d_kwidth', type=int, default=3,
                        help='2D kernel widths (Def: 3).')
    parser.add_argument('--SND', action='store_true', default=False)
    parser.add_argument('--g_aal', action='store_true', default=False)
    parser.add_argument('--g_aal_out', action='store_true', default=False)
    parser.add_argument('--skip_dropout', type=float, default=0)
    parser.add_argument('--d_dropout', type=float, default=0)
    parser.add_argument('--d_noise_std', type=float, default=1)
    parser.add_argument('--noise_dec_step', type=float, default=1e-3)
    parser.add_argument('--d_noise_epoch', type=int, default=3)
    parser.add_argument('--d_bnorm', action='store_true', default=False)
    parser.add_argument('--skip', action='store_true', default=False)
    parser.add_argument('--D_pool_type', type=str, default='conv',
                        help='Types: rnn, none, conv')
    parser.add_argument('--pesq_objective', action='store_true', default=False)
    parser.add_argument('--no_winit', action='store_true', default=False)
    parser.add_argument('--g_rnn_core', action='store_true', default=False,
                        help='Apply RNN pooling in G core')
    parser.add_argument('--D_pool_size', type=int, default=8)
    parser.add_argument('--max_ma', type=int, default=1000)
    parser.add_argument('--max_pad', type=int, default=160,
                        help='Translate intput to G randomly up to this num.')
    parser.add_argument('--tanh', action='store_true', default=False)
    parser.add_argument('--stereo_D', action='store_true', default=False,
                        help='Make comparative D between two channels '
                             'Original SEGAN behavior')
    parser.add_argument('--DG_tied', action='store_true', default=False,
                        help='Tie encoders from G to D')
    parser.add_argument('--linterp', action='store_true', default=False)
    parser.add_argument('--BID', action='store_true', default=False,
                        help='Bibranched Discriminator')
    parser.add_argument('--misalign_stereo', action='store_true', default=False)
    parser.add_argument('--no_z', action='store_true', default=False)
    parser.add_argument('--g_mlpconv', action='store_true', default=False)
    parser.add_argument('--g_subtract_mean', action='store_true', default=False)
    parser.add_argument('--g_onehot', action='store_true', default=False,
                        help='Apply onehot ID to g layers (only if spks '
                             'are loaded through spk2idx.')
    parser.add_argument('--bcgan', action='store_true', default=False,
                        help='Use classic loss BCE GAN (Def: False).')
    parser.add_argument('--l1_dec_epoch', type=int, default=10)
    parser.add_argument('--l1_weight', type=float, default=100,
                        help='L1 regularization weight (Def. 100). ')
    parser.add_argument('--d_real_weight', type=float, default=1)
    parser.add_argument('--d_fake_weight', type=float, default=1)
    parser.add_argument('--g_weight', type=float, default=1)
    parser.add_argument('--g_step_lr', type=int, default=None,
                        help='Update period in epochs for lr of G.')
    parser.add_argument('--max_ckpts', type=int, default=None)
    parser.add_argument('--d_step_lr', type=int, default=None,
                        help='Update period in epochs for lr of D.')
    parser.add_argument('--d_lr_gamma', type=float, default=0.5,
                        help='Mul decay factor for D')
    parser.add_argument('--g_lr_gamma', type=float, default=0.5,
                        help='Mul decay factor for G')
    parser.add_argument('--g_min_lr', type=float, default=0.0002)
    parser.add_argument('--d_min_lr', type=float, default=0.0002)
    parser.add_argument('--no_eval', action='store_true', default=False)
    parser.add_argument('--pooling_size', type=int, default=2,
                        help='Down/Up-sampling factor')
    parser.add_argument('--test_ckpt', type=str, default=None)
    parser.add_argument('--lbd', type=float, default=10,
                        help='Gradient penalty lambda for WSilentSEGAN ' \
                             '(Def: 10).')
    parser.add_argument('--spkid' ,type=int, default=None)
    parser.add_argument('--disc_type' ,type=str, default='vbnd')

    opts = parser.parse_args()

    opts.g_bias = not opts.no_g_bias
    opts.no_tanh = not opts.tanh

    if not os.path.exists(opts.synthesis_path):
        os.makedirs(opts.synthesis_path)
    
    print('Parsed arguments: ', json.dumps(vars(opts), indent=2))

    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    main(opts)
