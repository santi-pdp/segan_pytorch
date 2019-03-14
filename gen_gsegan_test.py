import argparse
from segan.datasets import SEOnlineDataset
from segan.transforms import *
import torch
import numpy as np
import random
import timeit
import json
import soundfile as sf
import os


def main(opts):
    # seed initialization
    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    # Build chunker transform with proper slice size
    #chunker = SingleChunkWav(opts.slice_size)
    chunker = None

    # Build transforms
    trans = PCompose([
        Resample(opts.resample_factors, report=True),
        Clipping(report=True),
        Chopper(max_chops=5, report=True),
    ], report=True)

    # create Dataset(s) and Dataloader(s)
    dset = SEOnlineDataset(opts.data_root,
                           distorteds=opts.distorted_roots,
                           chunker=chunker,
                           nsamples=0,
                           return_uttname=True,
                           transform=trans) 

    timings = []
    beg_t = timeit.default_timer()
    reports = {}
    for idx in range(len(dset)):
        uttname, clean, noisy, report = dset[idx]
        print(report)
        if len(report) == 0:
            report = None
        noisy = noisy[:clean.size(0)]
        assert clean.size() == noisy.size(), noisy.size()
        sf.write(os.path.join(opts.save_path, uttname),
                 noisy.data.numpy(), 16000)
        logname = os.path.splitext(uttname)[0] + '.json'
        with open(os.path.join(opts.log_save_path, logname), 'w') as log_f:
            log_f.write(json.dumps(report, indent=2))
        end_t = timeit.default_timer()
        timings.append(end_t - beg_t)
        mtime = np.mean(timings)
        beg_t = timeit.default_timer()
        if (idx + 1) % 50 == 0 or idx >= len(dset) - 1:
            print('Distorted file {:4d}/{:4d}, {}, mtime:{:.2f} s'.format(idx+1,
                                                                          len(dset),
                                                                          uttname,
                                                                          mtime),
                 end='\r')
    print('\nReport:')
    print(json.dumps(reports, indent=1))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str,
                        default="data_gsegan/distorted_testset_trimsil",
                        help="Path to save models (Def: seganv1_ckpt).")
    parser.add_argument('--data_root', type=str,
                        default='data_gsegan/clean_testset_trimsil')
    parser.add_argument('--seed', type=int, default=111, 
                        help="Random seed (Def: 111).")
    parser.add_argument('--distorted_roots', type=str, nargs='+',
                        default=['data_gsegan/whisper_testset_trimsil'])
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--slice_size', type=int, default=16384)
    parser.add_argument('--resample_factors', type=int, nargs='+',
                        default=[2, 4, 8])
    parser.add_argument('--log_save_path', type=str, default=None,
                        help='Path to store JSON logs of test generation.'
                             'If there is None, they will be saved in '
                             'save_path (Def: None).')
    parser.add_argument('--spks', type=str, nargs='+', default=None)

    opts = parser.parse_args()

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)
    if opts.log_save_path is None:
        opts.log_save_path = opts.save_path
    if not os.path.exists(opts.log_save_path):
        os.makedirs(opts.log_save_path)


    print('Parsed arguments: ', json.dumps(vars(opts), indent=2))
    main(opts)
