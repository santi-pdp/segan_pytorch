import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import *
from segan.datasets import *
import soundfile as sf
from scipy.io import wavfile
from torch.autograd import Variable
import multiprocessing as mp
import numpy as np
import tqdm
import random
import librosa
import matplotlib
import timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import glob
import os


class ArgParser(object):

    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

def process_utterance(uttname, model, save_path, preemph=0, 
                      device='cpu'):
    tbname = os.path.basename(uttname)
    wav, read = sf.read(uttname)
    if wav.shape[0] < 8000:
        return
    wav = pre_emphasize(wav, preemph)
    pwav = torch.FloatTensor(wav).view(1,1,-1)
    pwav = pwav.to(device)
    g_wav, g_c = model.generate(pwav)
    out_path = os.path.join(save_path, tbname) 
    sf.write(out_path, g_wav, 16000)

def main(opts):
    assert opts.cfg_file is not None
    assert opts.test_files is not None
    assert opts.g_pretrained_ckpt is not None

    with open(opts.cfg_file, 'r') as cfg_f:
        args = ArgParser(json.load(cfg_f))
        print('Loaded train config: ')
        print(json.dumps(vars(args), indent=2))
    args.cuda = opts.cuda
    if hasattr(args, 'wsegan') and args.wsegan:
        segan = WSEGAN(args)     
    else:
        segan = SEGAN(args)     
    segan.G.load_pretrained(opts.g_pretrained_ckpt, load_last=True)
    #if opts.cuda:
    #    segan.cuda()
    segan.G.eval()
    print(segan.G)
    print(segan.D)
    """
    for k, v in ckpt['state_dict'].items():
        print('||{} -> {}||'.format(k, v.size()))
    for k, v in dict(segan.G.named_parameters()).items():
        print('{} -> {}'.format(k, v.size()))
    """
    if opts.h5:
        with h5py.File(opts.test_files[0], 'r') as f:
            twavs = f['data'][:]
    else:
        # process every wav in the test_files
        if len(opts.test_files) == 1:
            # assume we read directory
            twavs = glob.glob(os.path.join(opts.test_files[0], '*.wav'))
        else:
            # assume we have list of files in input
            twavs = opts.test_files
    print('Cleaning {} wavs'.format(len(twavs)))
    for twav in tqdm.tqdm(twavs, total=len(twavs)):
        process_utterance(twav, segan, opts.synthesis_path,
                          args.preemph)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--g_pretrained_ckpt', type=str, default=None)
    parser.add_argument('--test_files', type=str, nargs='+', default=None)
    parser.add_argument('--h5', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--synthesis_path', type=str, default='segan_samples',
                        help='Path to save output samples (Def: ' \
                             'segan_samples).')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--soundfile', action='store_true', default=False)
    parser.add_argument('--cfg_file', type=str, default=None)

    opts = parser.parse_args()

    if not os.path.exists(opts.synthesis_path):
        os.makedirs(opts.synthesis_path)
    
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed_all(opts.seed)

    main(opts)
