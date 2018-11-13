import torch
import torch.nn as nn
import random
import torch.nn.utils as nnu
import torch.nn.functional as F
from collections import OrderedDict
try:
    from core import Model, LayerNorm
    from modules import *
except ImportError:
    from .core import Model, LayerNorm
    from .modules import *

# BEWARE: PyTorch >= 0.4.1 REQUIRED
from torch.nn.utils.spectral_norm import spectral_norm

#class BiDiscriminator(Model):
#    """ Branched discriminator for input and conditioner """
#    def __init__(self, fmaps, kwidth, activation,
#                 bnorm=False, pooling=2, SND=False, 
#                 dropout=0):
#        super().__init__(name='BiDiscriminator')
#        self.disc_in = nn.ModuleList()
#        self.disc_cond = nn.ModuleList()
#        for d_i, d_fmap in enumerate(fmaps):
#            if d_i == 0:
#                inp = 1
#            else:
#                inp = fmaps[d_i - 1]
#            self.disc_in.append(DiscBlock(inp, kwidth, d_fmap,
#                                          activation, bnorm,
#                                          pooling, SND, dropout))
#            self.disc_cond.append(DiscBlock(inp, kwidth, d_fmap,
#                                            activation, bnorm,
#                                            pooling, SND, dropout))
#        self.bili = nn.Linear(8 * fmaps[-1], 8 * fmaps[-1], bias=True)
#        if SND:
#            self.bili = spectral_norm(self.bili)
#
#    def forward(self, x):
#        x = torch.chunk(x, 2, dim=1)
#        hin = x[0]
#        hcond = x[1]
#        # store intermediate activations
#        int_act = {}
#        for ii, (in_layer, cond_layer) in enumerate(zip(self.disc_in,
#                                                        self.disc_cond)):
#            hin = in_layer(hin)
#            int_act['hin_{}'.format(ii)] = hin
#            hcond = cond_layer(hcond)
#            int_act['hcond_{}'.format(ii)] = hcond
#        hin = hin.view(hin.size(0), -1)
#        hcond = hcond.view(hin.size(0), -1)
#        bilinear_h = self.bili(hcond)
#        int_act['bilinear_h'] = bilinear_h
#        bilinear_out = torch.bmm(hin.unsqueeze(1),
#                                 bilinear_h.unsqueeze(2)).squeeze(-1)
#        norm1 = torch.norm(bilinear_h.data)
#        norm2 = torch.norm(hin.data)
#        bilinear_out = bilinear_out / max(norm1, norm2)
#        int_act['logit'] = bilinear_out
#        #return F.sigmoid(bilinear_out), bilinear_h, hin, int_act
#        return bilinear_out, bilinear_h, hin, int_act

class Discriminator(Model):
    
    def __init__(self, ninputs, fmaps,
                 kwidth, poolings,
                 pool_type='none',
                 pool_slen=None,
                 norm_type='bnorm',
                 bias=True,
                 phase_shift=None, 
                 sinc_conv=False):
        super().__init__(name='Discriminator')
        # phase_shift randomly occurs within D layers
        # as proposed in https://arxiv.org/pdf/1802.04208.pdf
        # phase shift has to be specified as an integer
        self.phase_shift = phase_shift
        if phase_shift is not None:
            assert isinstance(phase_shift, int), type(phase_shift)
            assert phase_shift > 1, phase_shift
        if pool_slen is None:
            raise ValueError('Please specify D network pool seq len '
                             '(pool_slen) in the end of the conv '
                             'stack: [inp_len // (total_pooling_factor)]')
        ninp = ninputs
        # SincNet as proposed in 
        # https://arxiv.org/abs/1808.00158
        if sinc_conv:
            # build sincnet module as first layer
            self.sinc_conv = SincConv(fmaps[0] // 2,
                                      251, 16e3, padding='SAME')
            inp = fmaps[0]
            fmaps = fmaps[1:]
        self.enc_blocks = nn.ModuleList()
        for pi, (fmap, pool) in enumerate(zip(fmaps,
                                              poolings),
                                          start=1):
            enc_block = GConv1DBlock(
                ninp, fmap, kwidth, stride=pool,
                bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap
        self.pool_type = pool_type
        if pool_type == 'none':
            # resize tensor to fit into FC directly
            pool_slen *= fmaps[-1]
            self.fc = nn.Sequential(
                nn.Linear(pool_slen, 256),
                nn.PReLU(256),
                nn.Linear(256, 128),
                nn.PReLU(128),
                nn.Linear(128, 1)
            )
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc[0])
                torch.nn.utils.spectral_norm(self.fc[2])
                torch.nn.utils.spectral_norm(self.fc[3])
        elif pool_type == 'conv':
            self.pool_conv = nn.Conv1d(fmaps[-1], 1, 1)
            self.fc = nn.Linear(pool_slen, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.pool_conv)
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gmax':
            self.gmax = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'gavg':
            self.gavg = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.fc)
        elif pool_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Conv1d(fmaps[-1], fmaps[-1], 1),
                nn.PReLU(fmaps[-1]),
                nn.Conv1d(fmaps[-1], 1, 1)
            )
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.mlp[0])
                torch.nn.utils.spectral_norm(self.mlp[1])
        else:
            raise TypeError('Unrecognized pool type: ', pool_type)
    
    def forward(self, x):
        h = x
        if hasattr(self, 'sinc_conv'):
            h_l, h_r = torch.chunk(h, 2, dim=1)
            h_l = self.sinc_conv(h_l)
            h_r = self.sinc_conv(h_r)
            h = torch.cat((h_l, h_r), dim=1)
        # store intermediate activations
        int_act = {}
        for ii, layer in enumerate(self.enc_blocks):
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
        if self.pool_type == 'conv':
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
            y = self.mlp(h)
        int_act['logit'] = y
        return y, int_act


if __name__ == '__main__':
    # pool_slen = 16 because we have input len 16384
    # and we perform 5 pooling layers of 4, so 16384 // (4 ** 5) = 16
    disc = Discriminator(2, [64, 128, 256, 512, 1024],
                         31, [4] * 5, pool_type='none',
                         pool_slen=16)
    print(disc)
    print('Num params: ', disc.get_n_params())
    x = torch.randn(1, 2, 16384)
    y, _ = disc(x)
    print(y)
    print('x size: {} -> y size: {}'.format(x.size(), y.size()))
