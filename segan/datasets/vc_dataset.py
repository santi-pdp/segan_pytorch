import torch
from torch.utils.data import Dataset
from scipy.io import wavfile
import glob
import os
try:
    from se_dataset import normalize_wave_minmax, pre_emphasize
except ImportError:
    from .se_dataset import normalize_wave_minmax, pre_emphasize


def varlen_wav_collate(batch):
    src_maxlen = 0
    trg_maxlen = 0
    for sample in batch:
        if len(sample) == 3:
            _, src, trg = sample
        else:
            src, trg = sample
        if src_maxlen < src.shape[0]:
            src_maxlen = src.shape[0]
        if trg_maxlen < trg.shape[0]:
            trg_maxlen = trg.shape[0]
    src_wav_b = torch.zeros(len(batch), 
                            src_maxlen)
    trg_wav_b = torch.zeros(len(batch), 
                            trg_maxlen)
    for bi, sample in enumerate(batch):
        if len(sample) == 3:
            _, src, trg = sample
        else:
            src, trg = sample
        src_wav_b[bi, :src.shape[0]] = torch.FloatTensor(src)
        trg_wav_b[bi, :trg.shape[0]] = torch.FloatTensor(trg)
    return '', src_wav_b, trg_wav_b

class VCDataset(Dataset):
    """
    At the moment JUST ONE-TO-ONE SPEAKER MAPPING
    """
    # TODO: EXTEND TO MULTI SPK LOAD
    def __init__(self, src_path, trg_path, preemph=0):
        super().__init__()
        self.src_path = src_path
        self.trg_path = trg_path
        self.preemph = preemph
        src_files = glob.glob(os.path.join(src_path, '*.wav'))
        trg_files = []
        for src_file in src_files:
            bname = os.path.basename(src_file)
            trg_file = os.path.join(trg_path, bname)
            assert os.path.exists(trg_file)
            trg_files.append(trg_file)
        self.src_files = src_files
        self.trg_files = trg_files

    def read_wav_file(self, wavfilename):
        rate, wav = wavfile.read(wavfilename)
        wav = normalize_wave_minmax(wav)
        wav = pre_emphasize(wav, self.preemph)
        return rate, wav

    def __getitem__(self, index):
        src_wav = self.read_wav_file(self.src_files[index])[1]
        trg_wav = self.read_wav_file(self.trg_files[index])[1]
        return src_wav, trg_wav

    def __len__(self):
        return len(self.src_files)

if __name__ == '__main__':
    dset = VCDataset('../../data/vc_data/trainset/VCC2SF1',
                     '../../data/vc_data/trainset/VCC2TM1')
    print(len(dset))
    print(dset.__getitem__(0))


