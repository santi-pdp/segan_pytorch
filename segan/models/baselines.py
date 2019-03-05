import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import timeit
import os
try:
    from .core import Model, Saver
except ImportError:
    from core import Model, Saver


class LPSRNN(Model):

    def __init__(self, lps_size=1025,
                 rnn_size=1500, rnn_layers=2):
        super().__init__()
        self.rnn = nn.GRU(lps_size, rnn_size, 
                          num_layers=rnn_layers,
                          bidirectional=True, 
                          batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * rnn_size, rnn_size),
            nn.PReLU(rnn_size, init=0.1),
            nn.Linear(rnn_size, lps_size)
        )

    def forward(self, x, state=None):
        h, state = self.rnn(x, state) 
        bsz, slen, rnn_feats = h.size()
        h = h.contiguous().view(bsz * slen, rnn_feats)
        y = self.mlp(h)
        y = y.view(bsz, slen, -1)
        return y, state

    def sample_dloader(self, dloader, device='cpu'):
        sample = next(dloader.__iter__())
        batch = sample
        if len(batch) == 2:
            clean, noisy = batch
            slice_idx = 0
            uttname = ''
        elif len(batch) == 3:
            uttname, clean, noisy = batch
            slice_idx = 0
        else:
            uttname, clean, noisy, slice_idx = batch
            slice_idx = slice_idx.to(device)
        clean = clean.to(device)
        noisy = noisy.to(device)
        return uttname, clean, noisy, slice_idx

    def split_stft(self, X):
        mag, pha = torch.chunk(X, 2, dim=1)
        return mag.squeeze(1), pha.squeeze(1)

    def train(self, opts, dloader, criterion, tr_samples,
              device='cpu'):
        print('Entering training loop')
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))
        self.optim = optim.Adam(self.parameters(), opts.lr)
        # Build savers for end of epoch, storing up to 3 epochs each
        eoe_saver = Saver(self, opts.save_path, max_ckpts=3,
                          optimizer=self.optim, prefix='EOE_G-')
        bpe = (tr_samples // opts.slice_size) // opts.batch_size \
                if tr_samples is not None else len(dloader)

        print('bpe: ', bpe)
        num_batches = len(dloader) 
        timings = []

        for iteration in range(1, opts.epoch * bpe + 1):
            beg_t = timeit.default_timer()
            uttname, clean, \
                    noisy, slice_idx = self.sample_dloader(dloader,
                                                           device)
            bsz = clean.size(0)
            cmag, cpha = self.split_stft(clean)
            nmag, npha = self.split_stft(noisy)
            ymag, state = self(cmag.transpose(1, 2).contiguous())
            ymag = ymag.transpose(1, 2)
            print('ymag size: ', ymag.size())
            raise NotImplementedError


if __name__ == '__main__':
    rnn = LPSRNN(1025)
    x = torch.randn(1, 100, 1025)
    y, _ = rnn(x)
    print('y size: ', y.size())
    print('Num params: ', rnn.get_n_params())
