from __future__ import print_function
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import gzip
import pickle
import timeit
import scipy.io.wavfile as wavfile
import numpy as np


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

def slice_signal_index(signal, window_size, stride):
    """ Slice input signal into indexes (beg, end) each

        # Arguments
            window_size: size of each slice
            stride: fraction of sliding window per window size

        # Returns
            A list of tuples (beg, end) sample indexes
    """
    assert stride <= 1, stride
    assert stride > 0, stride
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    slices = []
    offset = int(window_size * stride)
    #for beg_i in range(0, n_samples - (offset), offset):
    for beg_i in range(0, n_samples - window_size + 1, offset):
        end_i = beg_i + window_size
        if end_i > n_samples:
            # last slice is offset to past to fit full window
            beg_i = n_samples - window_size
            end_i = n_samples
        slice_ = (beg_i, end_i)
        slices.append(slice_)
    return slices

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
    def __init__(self, clean_dir, noisy_dir, preemph, cache_dir='.', slice_size=2**14,
                 stride = 0.5, max_samples=None, do_cache=False, verbose=False):
        super(SEDataset, self).__init__()
        self.clean_names = glob.glob(os.path.join(clean_dir, '*.wav'))
        self.noisy_names = glob.glob(os.path.join(noisy_dir, '*.wav'))
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
        self.verbose = verbose
        self.preemph = preemph
        if do_cache:
            self.read_wavs_and_cache()
        else:
            self.read_wavs()
        # make the slice indexes given slice_size and stride
        self.prepare_slicing()

    def read_wavs(self):
        self.clean_wavs = []
        self.noisy_wavs = []
        clen = len(self.clean_names)
        nlen = len(self.noisy_names)
        assert clen == nlen, clen
        if self.verbose:
            print('< Reading {} wav files... >'.format(clen))
        beg_t = timeit.default_timer()
        for i, (clean_name, noisy_name) in enumerate(zip(self.clean_names, self.noisy_names), start=1):
            # print('Reading wav pair {}/{}'.format(i, clen))
            c_rate, c_wav = wavfile.read(clean_name)
            if c_wav.shape[0] < self.slice_size:
                # skip this wav as it is shorter than the window
                continue
            if c_rate != 16e3:
                raise ValueError('Sampling rate is supposed to be 16.000 Hz')
            c_wav = normalize_wave_minmax(c_wav)
            c_wav = pre_emphasize(c_wav, self.preemph)
            self.clean_wavs.append(c_wav)

            n_rate, n_wav = wavfile.read(noisy_name)
            n_wav = normalize_wave_minmax(n_wav)
            n_wav = pre_emphasize(n_wav, self.preemph)
            self.noisy_wavs.append(n_wav)
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
        slicings = []
        verbose = self.verbose
        if verbose:
            print('< Slicing all signals with window'
                  ' {} and stride {}... >'.format(self.slice_size, self.stride))
        beg_t = timeit.default_timer()
        for w_i, (c_wav, n_wav) in enumerate(zip(self.clean_wavs,
                                                 self.noisy_wavs)):
            c_slices = slice_signal_index(c_wav, self.slice_size, self.stride)
            n_slices = slice_signal_index(n_wav, self.slice_size, self.stride)
            for c_slice, n_slice in zip(c_slices, n_slices):
                if verbose:
                    print('Id: {}, name: {}, c_slice: {}, n_slice: {}'.format(w_i, self.clean_names[w_i], c_slice,
                                                                              n_slice))
                slicings.append({'id':w_i, 'c_slice':c_slice,
                                 'n_slice':n_slice})
        self.slicings = slicings
        end_t = timeit.default_timer()
        if verbose:
            print('Sliced all signals in {} s'.format(end_t - beg_t))

    def extract_slice(self, index):
        slice_ = self.slicings[index]
        idx_, c_slice_, n_slice_ = slice_['id'], slice_['c_slice'], slice_['n_slice']
        c_signal = self.clean_wavs[idx_]
        n_signal = self.noisy_wavs[idx_]
        c_slice = c_signal[c_slice_[0]:c_slice_[1]]
        n_slice = n_signal[n_slice_[0]:n_slice_[1]]
        return c_slice, n_slice

    def __getitem__(self, index):
        c_slice, n_slice = self.extract_slice(index)
        # print('idx: {} c_slice shape: {}'.format(index, c_slice.shape))
        return torch.FloatTensor(c_slice), torch.FloatTensor(n_slice)

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
    dset = SEDataset('../../data/clean_trainset', '../../data/noisy_trainset', 0.95,
                      cache_dir=None, max_samples=100, verbose=True)
    sample_0 = dset.__getitem__(0)
    print('sample_0: ', sample_0)
