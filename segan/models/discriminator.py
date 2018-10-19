import torch
import torch.nn as nn
import random
import torch.nn.utils as nnu
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules import conv, Linear
try:
    from core import Model, LayerNorm, VirtualBatchNorm1d
    from modules import ResARModule
except ImportError:
    from .core import Model, LayerNorm, VirtualBatchNorm1d
    from .modules import ResARModule
from torch.nn.utils.spectral_norm import spectral_norm


class DiscBlock(nn.Module):

    def __init__(self, ninputs, kwidth, nfmaps,
                 activation, bnorm=False, pooling=2, SND=False, 
                 dropout=0):
        super().__init__()
        self.kwidth = kwidth
        self.pooling = pooling
        seq_dict = OrderedDict()
        self.conv = nn.Conv1d(ninputs, nfmaps, kwidth,
                              stride=pooling,
                              padding=0)
        if SND:
            self.conv = spectral_norm(self.conv)
        seq_dict['conv'] = conv
        if isinstance(activation, str):
            self.act = getattr(nn, activation)()
        else:
            self.act = activation
        self.bnorm = bnorm
        if bnorm:
            self.bn = nn.BatchNorm1d(nfmaps)
        self.dropout = dropout
        if dropout > 0:
            self.dout = nn.Dropout(dropout)

    def forward(self, x):
        if self.pooling == 1:
            x = F.pad(x, ((self.kwidth//2), self.kwidth//2))
        else:
            x = F.pad(x, ((self.kwidth//2)-1, self.kwidth//2))
        conv_h = self.conv(x)
        if self.bnorm:
            conv_h = self.bn(conv_h)
        conv_h = self.act(conv_h)
        if self.dropout:
            conv_h = self.dout(conv_h)
        return conv_h


class BiDiscriminator(Model):
    """ Branched discriminator for input and conditioner """
    def __init__(self, d_fmaps, kwidth, activation,
                 bnorm=False, pooling=2, SND=False, 
                 dropout=0):
        super().__init__(name='BiDiscriminator')
        self.disc_in = nn.ModuleList()
        self.disc_cond = nn.ModuleList()
        for d_i, d_fmap in enumerate(d_fmaps):
            if d_i == 0:
                inp = 1
            else:
                inp = d_fmaps[d_i - 1]
            self.disc_in.append(DiscBlock(inp, kwidth, d_fmap,
                                          activation, bnorm,
                                          pooling, SND, dropout))
            self.disc_cond.append(DiscBlock(inp, kwidth, d_fmap,
                                            activation, bnorm,
                                            pooling, SND, dropout))
        self.bili = nn.Linear(8 * d_fmaps[-1], 8 * d_fmaps[-1], bias=True)
        if SND:
            self.bili = spectral_norm(self.bili)

    def forward(self, x):
        x = torch.chunk(x, 2, dim=1)
        hin = x[0]
        hcond = x[1]
        # store intermediate activations
        int_act = {}
        for ii, (in_layer, cond_layer) in enumerate(zip(self.disc_in,
                                                        self.disc_cond)):
            hin = in_layer(hin)
            int_act['hin_{}'.format(ii)] = hin
            hcond = cond_layer(hcond)
            int_act['hcond_{}'.format(ii)] = hcond
        hin = hin.view(hin.size(0), -1)
        hcond = hcond.view(hin.size(0), -1)
        bilinear_h = self.bili(hcond)
        int_act['bilinear_h'] = bilinear_h
        bilinear_out = torch.bmm(hin.unsqueeze(1),
                                 bilinear_h.unsqueeze(2)).squeeze(-1)
        norm1 = torch.norm(bilinear_h.data)
        norm2 = torch.norm(hin.data)
        bilinear_out = bilinear_out / max(norm1, norm2)
        int_act['logit'] = bilinear_out
        #return F.sigmoid(bilinear_out), bilinear_h, hin, int_act
        return bilinear_out, bilinear_h, hin, int_act

class Discriminator(Model):
    
    def __init__(self, ninputs, d_fmaps, kwidth, activation,
                 bnorm=False, pooling=2, SND=False, pool_type='none',
                 dropout=0, Genc=None, pool_size=8, num_spks=None, 
                 phase_shift=None):
        super().__init__(name='Discriminator')
        # phase_shift randomly occurs within D layers
        # as proposed in https://arxiv.org/pdf/1802.04208.pdf
        # phase shift has to be specified as an integer
        self.phase_shift = phase_shift
        if phase_shift is not None:
            assert isinstance(phase_shift, int), type(phase_shift)
            assert phase_shift > 1, phase_shift
        if Genc is None:
            if not isinstance(activation, list):
                activation = [activation] * len(d_fmaps)
            if not isinstance(pooling, list):
                pooling = [pooling] * len(d_fmaps)
            else:
                assert len(pooling) == len(d_fmaps), len(pooling)
            self.disc = nn.ModuleList()
            for d_i, (d_fmap, pool) in enumerate(zip(d_fmaps, pooling)):
                act = activation[d_i]
                if d_i == 0:
                    inp = ninputs
                else:
                    inp = d_fmaps[d_i - 1]
                self.disc.append(DiscBlock(inp, kwidth, d_fmap,
                                           act, bnorm,
                                           pool, SND,
                                           dropout))
        else:
            print('Assigning Genc to D')
            # Genc and Denc MUST be same dimensions
            self.disc = Genc
        self.pool_type = pool_type
        if pool_type == 'none':
            # resize tensor to fit into FC directly
            pool_size *= d_fmaps[-1]
            if isinstance(act, nn.LeakyReLU):
                self.fc = nn.Sequential(
                    nn.Linear(pool_size, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 1)
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(pool_size, 256),
                    nn.PReLU(256),
                    nn.Linear(256, 128),
                    nn.PReLU(128),
                    nn.Linear(128, 1)
                )
        elif pool_type == 'rnn':
            if bnorm:
                self.ln = LayerNorm()
            pool_size = 128
            self.rnn = nn.LSTM(d_fmaps[-1], pool_size, batch_first=True,
                               bidirectional=True)
            # bidirectional size
            pool_size *= 2
            self.fc = nn.Linear(pool_size, 1)
        elif pool_type == 'conv':
            self.pool_conv = nn.Conv1d(d_fmaps[-1], 1, 1)
            self.fc = nn.Linear(pool_size, 1)
        elif pool_type == 'gmax':
            self.gmax = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(d_fmaps[-1], 1, 1)
        elif pool_type == 'gavg':
            self.gavg = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(d_fmaps[-1], 1, 1)
        elif pool_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(d_fmaps[-1], d_fmaps[-1]),
                nn.PReLU(d_fmaps[-1]),
                nn.Linear(d_fmaps[-1], 1)
            )
        else:
            raise TypeError('Unrecognized pool type: ', pool_type)
        outs = 1
        if num_spks is not None:
            outs += num_spks
    
    def forward(self, x):
        h = x
        # store intermediate activations
        int_act = {}
        for ii, layer in enumerate(self.disc):
            if self.phase_shift is not None:
                shift = random.randint(1, self.phase_shift)
                # 0.5 chance of shifting right or left
                right = random.random() > 0.5
                # split tensor in time dim (dim 2)
                if right:
                    sp1 = h[:, :, :-shift]
                    sp2 = h[:, :, -shift:]
                    h = torch.cat((sp2, sp1), dim=2)
                else:
                    sp1 = h[:, :, :shift]
                    sp2 = h[:, :, shift:]
                    h = torch.cat((sp2, sp1), dim=2)
            h = layer(h)
            int_act['h_{}'.format(ii)] = h
        if self.pool_type == 'rnn':
            if hasattr(self, 'ln'):
                h = self.ln(h)
                int_act['ln_conv'] = h
            ht, state = self.rnn(h.transpose(1,2))
            h = state[0]
            # concat both states (fwd, bwd)
            hfwd, hbwd = torch.chunk(h, 2, 0)
            h = torch.cat((hfwd, hbwd), dim=2)
            h = h.squeeze(0)
            int_act['rnn_h'] = h
            y = self.fc(h)
        elif self.pool_type == 'conv':
            h = self.pool_conv(h)
            h = h.view(h.size(0), -1)
            int_act['avg_conv_h'] = h
            y = self.fc(h)
        elif self.pool_type == 'none':
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gmax':
            h = self.gmax(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'gavg':
            h = self.gavg(h)
            h = h.view(h.size(0), -1)
            y = self.fc(h)
        elif self.pool_type == 'mlp':
            h = h.transpose(1, 2).contiguous()
            y = self.mlp(h)
        int_act['logit'] = y
        return y, int_act

class ARDiscriminator(Model):

    def __init__(self,
                 ninp=2,
                 dilations=[2, 4, 8, 16, 32, 64, 128, 256, 512],
                 kwidth=2,
                 fmaps=[256] * 9,
                 expansion_fmaps=128,
                 norm_type='snorm',
                 name='ARDiscriminator'):
        super().__init__(name=name)
        self.enc_blocks = nn.ModuleList()
        self.in_conv = nn.Conv1d(ninp, expansion_fmaps, 2)
        for pi, (fmap, dil) in enumerate(zip(fmaps,
                                             dilations),
                                         start=1):
            enc_block = ResARModule(expansion_fmaps, fmap,
                                    kwidth=kwidth,
                                    dilation=dil,
                                    norm_type=norm_type)
            self.enc_blocks.append(enc_block)

        self.mlp = nn.Sequential(
            nn.PReLU(expansion_fmaps, init=0),
            nn.Conv1d(expansion_fmaps, expansion_fmaps,
                      1),
            nn.PReLU(expansion_fmaps, init=0),
            nn.Conv1d(expansion_fmaps, 1,
                      1)
        )

    def forward(self, x):
        x_p = F.pad(x, (1, 0))
        h = self.in_conv(x_p)
        skip = None
        int_act = {'in_conv':h}
        for ei, enc_block in enumerate(self.enc_blocks):
            h = enc_block(h)
            if skip is None:
                skip = h
            else:
                skip += h
            int_act['skip_{}'.format(ei)] = h
        h = self.mlp(skip)
        int_act['logit'] = h
        return h, int_act


if __name__ == '__main__':
    #disc = Discriminator(2, [16, 32, 32, 64, 64, 128, 128, 256, 
    #                         256, 512, 1024], 31, 
    #                     nn.LeakyReLU(0.3))
    #disc = BiDiscriminator([16, 32, 32, 64, 64, 128, 128, 256, 
    #                         256, 512, 1024], 31, 
    #                     nn.LeakyReLU(0.3))
    disc = ARDiscriminator()
    print(disc)
    print(disc.num_parameters())
    from torch.autograd import Variable
    x = torch.randn(1, 2, 16384)
    y, _ = disc(x)
    print(y)
    print('x size: {} -> y size: {}'.format(x.size(), y.size()))
