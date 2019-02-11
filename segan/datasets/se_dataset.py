from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import os
import glob
import json
import gzip
import pickle
import timeit
import scipy.io.wavfile as wavfile
import numpy as np
import multiprocessing as mp
import random
import librosa
from ahoproc_tools.io import *
from ahoproc_tools.interpolate import *
import h5py


def collate_fn(batch):
    # first we have utt bname, then tensors
    data_batch = []
    uttname_batch = []
    for sample in batch:
        uttname_batch.append(sample[0])
        data_batch.append(sample[1:])
    data_batch = default_collate(data_batch)
    return [uttname_batch] + data_batch

def slice_signal(signal, window_sizes, stride=0.5):
    """ Slice input signal

        # Arguments
            window_sizes: list with different sizes to be sliced
            stride: fraction of sliding window per window size

        # Returns
            A list of numpy matrices, each one being of different window size
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    slices = []
    for window_size in window_sizes:
        offset = int(window_size * stride)
        slices.append([])
        for beg_i in range(n_samples + offset, offset):
            end_i = beg_i + offset
            if end_i > n_samples:
                # last slice is offset to past to fit full window
                beg_i = n_samples - offset
                end_i = n_samples
            slice_ = signal[beg_i:end_i]
            assert slice_.shape[0] == window_size, slice_.shape[0]
            slices[-1].append(slice_)
        slices[-1] = np.array(slices[-1], dtype=np.int32)
    return slices

def slice_index_helper(args):
    return slice_signal_index(*args)

def slice_signal_index(path, window_size, stride):
    """ Slice input signal into indexes (beg, end) each

        # Arguments
            window_size: size of each slice
            stride: fraction of sliding window per window size

        # Returns
            A list of tuples (beg, end) sample indexes
    """
    signal, rate = librosa.load(path, 16000)
    assert stride <= 1, stride
    assert stride > 0, stride
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    slices = []
    offset = int(window_size * stride)
    #for beg_i in range(0, n_samples - (offset), offset):
    for beg_i in range(0, n_samples - window_size + 1, offset):
        end_i = beg_i + window_size
        #if end_i >= n_samples:
            # last slice is offset to past to fit full window
        #    beg_i = n_samples - window_size
        #    end_i = n_samples
        slice_ = (beg_i, end_i)
        slices.append(slice_)
    return slices

def abs_normalize_wave_minmax(x):
    x = x.astype(np.int32)
    imax = np.max(np.abs(x))
    x_n = x / imax
    return x_n 

def abs_short_normalize_wave_minmax(x):
    imax = 32767.
    x_n = x / imax
    return x_n 

def dynamic_normalize_wave_minmax(x):
    x = x.astype(np.int32)
    imax = np.max(x)
    imin = np.min(x)
    x_n = (x - np.min(x)) / (float(imax) - float(imin))
    return x_n * 2 - 1

def normalize_wave_minmax(x):
    return (2./65535.) * (x - 32767.) + 1.

def pre_emphasize(x, coef=0.95):
    if coef <= 0:
        return x
    x0 = np.reshape(x[0], (1,))
    diff = x[1:] - coef * x[:-1]
    concat = np.concatenate((x0, diff), axis=0)
    return concat

def de_emphasize(y, coef=0.95):
    if coef <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coef * x[n - 1] + y[n]
    return x

class SEDataset(Dataset):
    """ Speech enhancement dataset """
    def __init__(self, clean_dir, noisy_dir, preemph, cache_dir='.', 
                 split='train', slice_size=2**14,
                 stride = 0.5, max_samples=None, do_cache=False, verbose=False,
                 slice_workers=2, preemph_norm=False,
                 random_scale=[1]):
        super(SEDataset, self).__init__()
        print('Creating {} split out of data in {}'.format(split, clean_dir))
        self.clean_names = glob.glob(os.path.join(clean_dir, '*.wav'))
        self.noisy_names = glob.glob(os.path.join(noisy_dir, '*.wav'))
        print('Found {} clean names and {} noisy'
              ' names'.format(len(self.clean_names), len(self.noisy_names)))
        self.slice_workers = slice_workers
        if len(self.clean_names) != len(self.noisy_names) or \
           len(self.clean_names) == 0:
            raise ValueError('No wav data found! Check your data path please')
        if max_samples is not None:
            assert isinstance(max_samples, int), type(max_samples)
            self.clean_names = self.clean_names[:max_samples]
            self.noisy_names = self.noisy_names[:max_samples]
        # path to store pairs of wavs
        self.cache_dir = cache_dir
        self.slice_size = slice_size
        self.stride = stride
        self.split = split
        self.verbose = verbose
        self.preemph = preemph
        # order is preemph + norm (rather than norm + preemph)
        self.preemph_norm = preemph_norm
        # random scaling list, selected per utterance
        self.random_scale = random_scale
        #self.read_wavs()
        cache_path = cache_dir#os.path.join(cache_dir, '{}_chunks.pkl'.format(split))
        #if os.path.exists(cache_path):
        #    with open(cache_path ,'rb') as ch_f:
        #        self.slicings = pickle.load(ch_f)
        #else:
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if not os.path.exists(os.path.join(cache_path,
                                           '{}_idx2slice.pkl'.format(split))):
            # make the slice indexes given slice_size and stride
            self.prepare_slicing()
            #with open(os.path.join(cache_path, '{}_cache.cfg'.format(split)), 'w') as cfg_f:
            #    cfg_f.write(json.dumps({'num_slicings':len(self.slicings)}))
            with open(os.path.join(cache_path, '{}_idx2slice.pkl'.format(split)), 'wb') as i2s_f:
                pickle.dump(self.idx2slice, i2s_f)
            #if do_cache:
            for s_i, slicing in self.slicings.items():
                with open(os.path.join(cache_path, '{}_{}.pkl'.format(split, s_i)), 'wb') as ch_f:
                    # store slicing results
                    pickle.dump(slicing, ch_f)
            self.num_samples = len(self.idx2slice)
            self.slicings = None
        else:
            #with open(os.path.join(cache_path, '{}_cache.cfg'.format(split)), 'r') as cfg_f:
            #    self.num_samples = json.load(cfg_f)
            with open(os.path.join(cache_path, '{}_idx2slice.pkl'.format(split)), 'rb') as i2s_f:
                self.idx2slice = pickle.load(i2s_f)
            print('Loaded {} idx2slice items'.format(len(self.idx2slice)))

    def read_wav_file(self, wavfilename):
        rate, wav = wavfile.read(wavfilename)
        if self.preemph_norm:
            wav = pre_emphasize(wav, self.preemph)
            wav = normalize_wave_minmax(wav)
        else:
            wav = normalize_wave_minmax(wav)
            wav = pre_emphasize(wav, self.preemph)
        return rate, wav

    def read_wavs(self):
        #self.clean_wavs = []
        self.clean_paths = []
        #self.noisy_wavs = []
        self.noisy_paths = []
        clen = len(self.clean_names)
        nlen = len(self.noisy_names)
        assert clen == nlen, clen
        if self.verbose:
            print('< Reading {} wav files... >'.format(clen))
        beg_t = timeit.default_timer()
        for i, (clean_name, noisy_name) in enumerate(zip(self.clean_names, self.noisy_names), start=1):
            # print('Reading wav pair {}/{}'.format(i, clen))
            #c_rate, c_wav = self.read_wav_file(clean_name)
            #if c_wav.shape[0] < self.slice_size:
                # skip this wav as it is shorter than the window
            #    continue
            #if c_rate != 16e3:
            #    raise ValueError('Sampling rate is supposed to be 16.000 Hz')
            #self.clean_wavs.append(c_wav)
            self.clean_paths.append(clean_name)

            #n_rate, n_wav = self.read_wav_file(noisy_name)
            #self.noisy_wavs.append(n_wav)
            self.noisy_paths.append(noisy_name)
        end_t = timeit.default_timer()
        if self.verbose:
            print('> Loaded files in {} s <'.format(end_t - beg_t))

    def read_wavs_and_cache(self):
        """ Read in all clean and noisy wavs """
        cache_path = os.path.join(self.cache_dir, 'cached_pair.pkl')
        try:
            with open(cache_path) as f_in:
                cache = pickle.load(f_in)
                if self.verbose:
                    print('Reading clean and wav pair from ', cache_path)
                self.clean_wavs = cache['clean']
                self.noisy_wavs = cache['noisy']
        except IOError:
            self.read_wavs()
            cache = {'noisy':self.noisy_wavs, 'clean':self.clean_wavs}
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            with open(cache_path, 'wb') as f_out:
                pickle.dump(cache, f_out)
                if self.verbose:
                    print('Cached clean and wav pair into ', cache_path)

    def prepare_slicing(self):
        """ Make a dictionary containing, for every wav file, its
            slices performed sequentially in steps of stride and
            sized slice_size
        """
        slicings = {}
        idx2slice = []
        verbose = self.verbose
        if verbose:
            print('< Slicing all signals with window'
                  ' {} and stride {}... >'.format(self.slice_size, self.stride))
        beg_t = timeit.default_timer()
        pool = mp.Pool(self.slice_workers)
        clean_args = [(self.clean_names[i], self.slice_size, self.stride) for \
                      i in range(len(self.clean_names))]
        c_slices = pool.map(slice_index_helper, clean_args)
        noisy_args = [(self.noisy_names[i], self.slice_size, self.stride) for \
                      i in range(len(self.noisy_names))]
        n_slices = pool.map(slice_index_helper, noisy_args)
        if len(n_slices) != len(c_slices):
            raise ValueError('n_slices and c_slices have different lengths:'
                             '{} != {}'.format(len(n_slices), len(c_slices)))
        for w_i, (c_slice, n_slice) in enumerate(zip(c_slices, n_slices)):
            c_path = self.clean_names[w_i]
            n_path = self.noisy_names[w_i]
            if w_i not in slicings:
                slicings[w_i] = []
            for t_i, (c_ss, n_ss) in enumerate(zip(c_slice, n_slice)):
                if c_ss[1] - c_ss[0] < 1024:
                    # decimate less than 4096 samples window
                    continue
                slicings[w_i].append({'c_slice':c_ss,
                                      'n_slice':n_ss, 'c_path':c_path,
                                      'n_path':n_path,
                                      'slice_idx':t_i})
                idx2slice.append((w_i, t_i))
        """
        for w_i, (c_path, n_path) in enumerate(zip(self.clean_names,
                                                   self.noisy_names)):
            c_wav, rate = librosa.load(c_path)
            n_wav, rate = librosa.load(n_path)
            c_slices = slice_signal_index(c_wav, self.slice_size, self.stride)
            n_slices = slice_signal_index(n_wav, self.slice_size, self.stride)
            for c_slice, n_slice in zip(c_slices, n_slices):
                if c_slice[1] - c_slice[0] < 4096:
                    continue
                if verbose:
                    print('Id: {}, name: {}, c_slice: {}, n_slice: {}'.format(w_i, self.clean_names[w_i], c_slice,
                                                                              n_slice))
                slicings.append({'id':w_i, 'c_slice':c_slice,
                                 'n_slice':n_slice,
                                 'c_path':c_path,
                                 'n_path':n_path})
        """
        self.slicings = slicings
        self.idx2slice = idx2slice
        end_t = timeit.default_timer()
        if verbose:
            print('Sliced all signals in {} s'.format(end_t - beg_t))

    def extract_slice(self, index):
        # load slice
        s_i, e_i = self.idx2slice[index]
        #print('selected item: ', s_i, e_i)
        slice_file = os.path.join(self.cache_dir,
                                  '{}_{}.pkl'.format(self.split, s_i))
        #print('reading slice file: ', slice_file)
        with open(slice_file, 'rb') as s_f:
            slice_ = pickle.load(s_f)
            #print('slice_: ', slice_)
            slice_ = slice_[e_i]
            c_slice_, n_slice_ = slice_['c_slice'], slice_['n_slice']
            slice_idx = slice_['slice_idx']
            n_path = slice_['n_path']
            bname = os.path.splitext(os.path.basename(n_path))[0]
            met_path = os.path.join(os.path.dirname(n_path), 
                                    bname + '.met')
            ssnr = None
            pesq = None
            if os.path.exists(met_path):
                metrics = json.load(open(met_path, 'r'))
                pesq = metrics['pesq']
                ssnr = metrics['ssnr']
            #c_signal, rate = librosa.load(slice_['c_path'])
            #n_signal, rate = librosa.load(slice_['n_path'])
            c_signal = self.read_wav_file(slice_['c_path'])[1]
            n_signal = self.read_wav_file(slice_['n_path'])[1]
            #c_signal = self.clean_wavs[idx_]
            #n_signal = self.noisy_wavs[idx_]
            c_slice = c_signal[c_slice_[0]:c_slice_[1]]
            n_slice = n_signal[n_slice_[0]:n_slice_[1]]
            if n_slice.shape[0] > c_slice.shape[0]:
                n_slice = n_slice[:c_slice.shape[0]]
            if c_slice.shape[0] > n_slice.shape[0]:
                c_slice = c_slice[:n_slice.shape[0]]
            #print('c_slice[0]: {} c_slice[1]: {}'.format(c_slice_[0],
            #                                             c_slice_[1]))
            if c_slice.shape[0] < self.slice_size:
                pad_t = np.zeros((self.slice_size - c_slice.shape[0],))
                c_slice = np.concatenate((c_slice, pad_t))
                n_slice = np.concatenate((n_slice, pad_t))
            #print('c_slice shape: ', c_slice.shape)
            #print('n_slice shape: ', n_slice.shape)
            bname = os.path.splitext(os.path.basename(n_path))[0]
            return c_slice, n_slice, pesq, ssnr, slice_idx, bname

    def __getitem__(self, index):
        c_slice, n_slice, pesq, ssnr, slice_idx, bname = self.extract_slice(index)
        rscale = random.choice(self.random_scale)
        if rscale != 1:
            c_slice = rscale * c_slice
            n_slice = rscale * n_slice
        returns = [bname, torch.FloatTensor(c_slice), torch.FloatTensor(n_slice),
                  slice_idx]
        if pesq is not None:
            returns.append(torch.FloatTensor([pesq]))
        if ssnr is not None:
            returns.append(torch.FloatTensor([ssnr]))
        # print('idx: {} c_slice shape: {}'.format(index, c_slice.shape))
        return returns

    def __len__(self):
        return len(self.idx2slice)

class RandomChunkSEDataset(Dataset):
    """ Random Chunking Speech enhancement dataset """
    def __init__(self, clean_dir, noisy_dir, preemph, 
                 split='train', slice_size=2**14,
                 max_samples=None, utt2spk=None, spk2idx=None):
        super(RandomChunkSEDataset, self).__init__()
        print('Creating {} split out of data in {}'.format(split, clean_dir))
        self.preemph = preemph
        # file containing pointers: baename (no ext) --> spkid
        self.utt2spk = utt2spk
        # dict containing mapping spkid --> int idx
        self.spk2idx = spk2idx
        if self.utt2spk is not None and self.spk2idx is None:
            raise ValueError('Please specify spk2idx too with utt2spk!')
        if utt2spk is not None:
            self.read_utt2spk()
        self.samples = {}
        self.slice_size = slice_size
        self.clean_names = glob.glob(os.path.join(clean_dir, '*.wav'))
        for c_i, cname in enumerate(self.clean_names):
            bname = os.path.basename(cname)
            self.samples[c_i] = {'clean':cname,
                                 'noisy':os.path.join(noisy_dir, bname)}

    def read_utt2spk(self):
        utt2spk = {}
        with open(self.utt2spk, 'r') as utt_f:
            for line in utt_f:
                line = line.rstrip().split('\t')
                uttname = os.path.splitext(os.path.basename(line[0]))[0]
                utt2spk[uttname] = line[1]
        self.utt2spk = utt2spk

    def read_wav_file(self, wavfilename):
        #rate, wav = wavfile.read(wavfilename)
        wav, rate = librosa.load(wavfilename, 16000)

        #wav = abs_short_normalize_wave_minmax(wav)
        wav = pre_emphasize(wav, self.preemph)
        return rate, wav

    def __getitem__(self, index):
        sample = self.samples[index]
        cpath = sample['clean']
        bname = os.path.splitext(os.path.basename(cpath))[0]
        npath = sample['noisy']
        returns = [bname]
        # slice them randomly
        cwav = self.read_wav_file(cpath)[1]
        nwav = self.read_wav_file(npath)[1]
        min_L = min(cwav.shape[0], nwav.shape[0])
        if self.slice_size > min_L:
            slice_size = min_L
        else:
            slice_size = self.slice_size
        slice_idx = random.randint(0, min_L - slice_size)
        cslice = cwav[slice_idx:slice_idx + self.slice_size] 
        nslice = nwav[slice_idx:slice_idx + self.slice_size] 
        if min_L < self.slice_size:
            c_pad_size = self.slice_size - cslice.shape[0]
            n_pad_size = self.slice_size - nslice.shape[0]
            c_pad_T = np.zeros(c_pad_size,)
            n_pad_T = np.zeros(n_pad_size,)
            # pad to desired size
            cslice  = np.concatenate((cslice, c_pad_T), axis=0)
            nslice  = np.concatenate((nslice, n_pad_T), axis=0)
        returns += [torch.FloatTensor(cslice), 
                    torch.FloatTensor(nslice)]
        if self.utt2spk is not None:
            spk = self.utt2spk[bname]
            spkidx = self.spk2idx[spk]
            returns.append(torch.LongTensor([spkidx]))
        return returns

    def __len__(self):
        return len(self.samples)

class RandomChunkSEF0Dataset(Dataset):
    """ Random Chunking Speech enhancement dataset loading
        F0 curves from aco path rather than wavs """
    def __init__(self, clean_dir, noisy_dir, preemph=0, 
                 split='train', slice_size=2**14,
                 max_samples=None):
        super(RandomChunkSEF0Dataset, self).__init__()
        print('Creating {} split out of data in {}'.format(split, clean_dir))
        self.preemph = preemph
        # file containing pointers: baename (no ext) --> spkid
        # dict containing mapping spkid --> int idx
        self.samples = {}
        self.slice_size = slice_size
        self.clean_names = glob.glob(os.path.join(clean_dir, '*.wav'))
        for c_i, cname in enumerate(self.clean_names):
            bname = os.path.splitext(os.path.basename(cname))[0]
            self.samples[c_i] = {'clean':cname,
                                 'noisy':os.path.join(noisy_dir, bname) + \
                                 '.lf0'}

    def read_wav_file(self, wavfilename):
        rate, wav = wavfile.read(wavfilename)
        wav = pre_emphasize(wav, self.preemph)
        #wav = dynamic_normalize_wave_minmax(wav)
        wav = abs_normalize_wave_minmax(wav)
        return rate, wav

    def __getitem__(self, index):
        sample = self.samples[index]
        cpath = sample['clean']
        bname = os.path.splitext(os.path.basename(cpath))[0]
        npath = sample['noisy']
        returns = [bname]
        # slice them randomly
        cwav = self.read_wav_file(cpath)[1]
        lf0 = read_aco_file(npath)
        ilf0, uv = interpolation(lf0, -10000000000)
        ilf0[ilf0 < -1000] = np.log(60)
        # append zeros in the end to show EOS
        ilf0 = np.concatenate((ilf0, np.zeros((1,))), axis=0)
        uv = np.concatenate((uv, np.zeros((1,))), axis=0)
        min_L = cwav.shape[0]
        #min_L = lf0.shape[0] * 80
        #print('cwav shape: ', cwav.shape)
        cwav = cwav[:min_L]
        #print('trimmed cwav shape: ', cwav.shape)
        if self.slice_size > min_L:
            slice_size = min_L
        else:
            slice_size = self.slice_size
        slice_idx = random.randint(0, min_L - slice_size)
        cslice = cwav[slice_idx:slice_idx + self.slice_size] 
        #print('slice_idx: ', slice_idx)
        #print('slice_idx // 80: ', slice_idx // 80)
        if slice_size < self.slice_size:
            print('WARNING: cwav shape: ', cwav.shape[0])
        lf0slice = np.zeros(((self.slice_size // 80) + 1,))
        uvslice = np.zeros(((self.slice_size // 80) + 1,))
        ilf0_s = ilf0[(slice_idx // 80):(slice_idx // 80) + \
                      (self.slice_size // 80) + 1]
        uv_s = uv[(slice_idx // 80):(slice_idx // 80) + \
                  (self.slice_size // 80) + 1]
        lf0slice[:ilf0_s.shape[0]] = ilf0_s
        uvslice[:uv_s.shape[0]] = uv_s
        if min_L < self.slice_size:
            c_pad_size = self.slice_size - cslice.shape[0]
            c_pad_T = np.zeros(c_pad_size,)
            # pad to desired size
            cslice  = np.concatenate((cslice, c_pad_T), axis=0)
        returns += [torch.FloatTensor(cslice), 
                    torch.FloatTensor(lf0slice),
                    torch.FloatTensor(uvslice)]
        return returns

    def __len__(self):
        return len(self.samples)

class SEH5Dataset(Dataset):
    """ Speech enhancement dataset from H5 data file. 
        The pairs must be named (data, label), being each
        one a dataset containing wav chunks (already chunked
        to fixed size).
    """
    def __init__(self, data_root, split, preemph, 
                 verbose=False,
                 preemph_norm=False,
                 random_scale=[1]):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.preemph = preemph
        self.verbose = verbose
        self.random_scale = random_scale
        h5_file = os.path.join(data_root, split + '.h5')
        if not os.path.exists(h5_file):
            raise FileNotFoundError(h5_file)
        f = h5py.File(h5_file, 'r')
        ks = list(f.keys())
        assert 'data' in ks, ks
        assert 'label' in ks, ks
        if verbose:
            print('Found H5 file {} with {} samples'.format(h5_file,
                                                            f['data'].shape[0]))
        self.f = f

    def __getitem__(self, index):
        c_slice = self.f['data'][index]
        n_slice = self.f['label'][index]
        rscale = random.choice(self.random_scale)
        if rscale != 1:
            c_slice = rscale * c_slice
            n_slice = rscale * n_slice
        # uttname not known with H5
        returns = ['N/A', torch.FloatTensor(c_slice).squeeze(-1), 
                   torch.FloatTensor(n_slice).squeeze(-1), 0]
        return returns

    def __len__(self):
        return self.f['data'].shape[0]

if __name__ == '__main__':
    #dset = SEDataset('../../data/clean_trainset', '../../data/noisy_trainset', 0.95,
    #                  cache_dir=None, max_samples=100, verbose=True)
    #sample_0 = dset.__getitem__(0)
    #print('sample_0: ', sample_0)

    #dset = RandomChunkSEF0Dataset('../../data/silent/clean_trainset',
    #                              '../../data/silent/lf0_trainset', 0.)
    #sample_0 = dset.__getitem__(0)
    #print('len sample_0: ', len(sample_0))
    dset = SEH5Dataset('../../data/widebandnet_h5/speaker1', 'train',
                       0.95, verbose=True)
    print(len(dset))
