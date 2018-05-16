import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import *
from segan.datasets import *
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
    if opts.cuda:
        segan.cuda()
    if opts.mode == 'train':
        print('Inside train mode...')
        # create dataset and dataloader
        dset = SEDataset(opts.clean_trainset, 
                         opts.noisy_trainset, 
                         opts.preemph,
                         do_cache=True,
                         cache_dir=opts.cache_dir,
                         split='train',
                         stride=0.5,
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
                            stride=0.5,
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
                    #va_dloader=None)
                    va_dloader=va_dloader)

    if opts.mode == 'test':
        raise NotImplementedError

    if opts.mode == 'filter_analysis':
        raise NotImplementedError
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.fftpack import fft
        segan.load(opts.save_path)
        segan.G.eval()
        print(opts.test_file)
        if opts.test_file is not None:
            rate, wav = wavfile.read(opts.test_file)
            #wav = normalize_wave_minmax(wav)
            #wav = dynamic_normalize_wave_minmax(wav)
            wav = abs_normalize_wave_minmax(wav)
            print('in wav min:{}, max:{}'.format(wav.min(), wav.max()))
            wav = torch.FloatTensor(wav).view(1,-1,1)
            wav = make_divN(wav, 2048)
            wav = Variable(wav).view(1,1,-1)
            if opts.cuda:
                wav = wav.cuda()
            print('in wav size: ', wav.size())
            # infer waveforms at different layer levels and check what is in them
            spkid = opts.spkid
            if opts.spkid is not None:
                spkid = Variable(torch.LongTensor([[opts.spkid]]))
            g_wav, hid_wav = segan.generate(wav, ret_hid=True, spkid=spkid)
            wavfile.write('gen-spk{}.wav'.format(opts.spkid), 16000,
                          g_wav.squeeze().cpu().data.numpy())
            print('num of hid_wavs: ', len(hid_wav)) 
            print('Generated wav: ', g_wav.size())
            for k, hw in hid_wav.items():
                print('FFTing feats of layer:{} with size:{}'.format(k,
                                                                     hw.size()))
                num_feats = hw.size(1)
                for nf in range(num_feats):
                    f_wav = hw[0, nf, :].cpu().data.numpy()
                    show_spec(f_wav)
                    plt.savefig(os.path.join(opts.save_path,
                                             '{}-{}_spec.png'.format(k,nf)),
                                dpi=300)
                    plt.close()

        print('Inside filter_analysis mode...')
        g_enc_blocks = segan.G.gen_enc
        g_enc_blocks_d = dict(g_enc_blocks.named_parameters())
        g_enc_num_weights = sum(1 for k, v in g_enc_blocks_d.items() if \
                                'weight' in k and 'act' not in k)
        g_dec_blocks = segan.G.gen_dec
        W = g_enc_blocks[0].conv.weight
        print('W size: ', W.size())
        # Analyze Generator Encoder weights
        plt.figure(figsize=(40, 20))
        plt.title('SEGAN response filters')
        MAX_PLOT=16
        total_plots = 1
        NFFT = 128
        fs = 16000 * np.array(list(range(NFFT))) / NFFT
        for l_i, (param_k, param_v) in enumerate(g_enc_blocks_d.items(), 
                                                 start=1):
            if 'weight' in param_k and 'act' not in param_k:
                # Get shape of weight tensor [out_ch, in_ch, width]
                out_chs, in_chs, width = param_v.size()
                print('Weight: {} --> Size: {}'.format(param_k,
                                                       param_v.size()))
                for out_ch in range(1, min(MAX_PLOT + 1, out_chs + 1)):
                    plt.subplot(g_enc_num_weights, MAX_PLOT, total_plots)
                    # k-th weight averaged in channels, plot in width
                    kw_avg = np.mean(param_v[out_ch - 1, :, :].cpu().data.numpy(), 
                                     axis=0)
                    KW_avg_F = fft(kw_avg, n=NFFT)[:NFFT//2]
                    #plt.stem(kw_avg)
                    plt.plot(fs[:NFFT//2], np.abs(KW_avg_F), linewidth=2.5)
                    plt.xlabel('[Hz]')
                    total_plots += 1
                #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                plt.tight_layout()
                plt.savefig(os.path.join(opts.save_path, 'filters.png'), dpi=300)
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spk2idx', type=str, default=None)
    parser.add_argument('--utt2spk', type=str, default=None)
    parser.add_argument('--rc_dataset', action='store_true', 
                        default=False,
                        help='Use the random chunker Dataset')
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
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
    parser.add_argument('--patience', type=int, default=20)
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
    parser.add_argument('--save_path', type=str, default="seganv1_ckpt",
                        help="Path to save models (Def: seganv1_ckpt).")
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
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--test_clean_dir', type=str, default=None)
    parser.add_argument('--synthesis_path', type=str, default='segan_samples',
                        help='Path to save output samples (Def: ' \
                             'segan_samples).')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max num of samples to train (Def: None).')
    parser.add_argument('--g_act', type=str, default='prelu')
    parser.add_argument('--d_act', type=str, default='prelu')
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
    parser.add_argument('--eval_workers', type=int, default=10)
    parser.add_argument('--slice_workers', type=int, default=2)
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
    parser.add_argument('--d_updates', type=int, default=1)
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
    parser.add_argument('--l1_dec_epoch', type=int, default=100)
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

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    if opts.mode == 'train':
        # save opts
        with open(os.path.join(opts.save_path, 'train.opts'), 'w') as cfg_f:
            cfg_f.write(json.dumps(vars(opts), indent=2))

    elif opts.mode == 'test':
        if not os.path.exists(opts.synthesis_path):
            os.makedirs(opts.synthesis_path)
        
        # load train config
        with open(os.path.join(opts.save_path, 'train.opts'), 'r') as cfg_f:
            cfg = json.load(cfg_f)
            for k, v in cfg.items():
                if hasattr(opts, k) and k != 'mode' and k != 'save_path' and \
                   k != 'synthesis_path' and k != 'test_dir' and k != 'seed' \
                   and k != 'test_clean_dir' and k != 'batch_size':
                    setattr(opts, k, v)
            # ensure no pretrained available
            opts.pretrained_ckpt = None

    elif opts.mode == 'filter_analysis':
        # load train config
        with open(os.path.join(opts.save_path, 'train.opts'), 'r') as cfg_f:
            cfg = json.load(cfg_f)
            for k, v in cfg.items():
                if hasattr(opts, k) and k != 'mode' and k != 'save_path' and \
                   k != 'test_file' and k != 'seed':
                    setattr(opts, k, v)
            # ensure no pretrained available
            opts.pretrained_ckpt = None
            
    print('Parsed arguments: ', json.dumps(vars(opts), indent=2))

    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    main(opts)
