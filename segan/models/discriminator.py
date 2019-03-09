import torch
import torch.nn as nn
import random
import torch.nn.utils as nnu
import torch.nn.functional as F
from collections import OrderedDict
from waveminionet.models.frontend import *
try:
    from core import Model, LayerNorm
    from modules import *
except ImportError:
    from .core import Model, LayerNorm
    from .modules import *

# BEWARE: PyTorch >= 0.4.1 REQUIRED
from torch.nn.utils.spectral_norm import spectral_norm


class DiscriminatorFE(Model):

    def __init__(self, 
                 fmaps=[128 ,128, 128, 128],
                 poolings=[2, 4, 4, 5],
                 kwidths=[22, 44, 44, 55],
                 frontend=None,
                 nheads=8,
                 hidden_size=128,
                 ff_size=128,
                 ft_fe=False,
                 bias=False,
                 norm_type='inorm',
                 pool_type='mlp',
                 phase_shift=None,
                 name='DiscriminatorFE'):
        super().__init__(name=name)
        self.phase_shift = phase_shift
        if frontend is None:
            self.frontend = WaveFe()
        else:
            self.frontend = frontend
        self.ft_fe = ft_fe
        emb_dim = self.frontend.emb_dim
        # ---------------------------------
        # Build Encoder for G(e)
        ninp = 1
        self.enc_blocks = nn.ModuleList()
        for pi, (fmap, pool, kw) in enumerate(zip(fmaps,
                                                  poolings,
                                                  kwidths),
                                              start=1):
            enc_block = GConv1DBlock(
                ninp, fmap, kw, stride=pool,
                bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap

        # build Multi-Head Attention
        if pool_type == 'mha':
            # pre-project noisy and c_e for MHA
            self.W_noisy = nn.Conv1d(emb_dim, hidden_size, 1)
            self.W_ce = nn.Conv1d(ninp, hidden_size, 1)
            self.mha = MultiHeadAttention(nheads, hidden_size)
            self.mha_norm = nn.InstanceNorm1d(hidden_size)
            self.mlp = nn.Sequential(
                nn.Conv1d(hidden_size, ff_size, 1),
                nn.InstanceNorm1d(ff_size),
                nn.PReLU(ff_size),
                nn.Conv1d(ff_size, 1, 1)
            )
        elif pool_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(50 * (ninp + emb_dim), ff_size),
                nn.BatchNorm1d(ff_size),
                nn.PReLU(ff_size),
                nn.Linear(ff_size, 1)
            )
        elif pool_type == 'gl':
            # Global and local mlp poolings
            self.mlp = nn.Sequential(
                nn.Linear(50 * 2 * ninp, ff_size),
                nn.BatchNorm1d(ff_size),
                nn.PReLU(ff_size),
                nn.Linear(ff_size, 1)
            )
            self.mlp_l = nn.Sequential(
                nn.Conv1d(2 * ninp, ff_size, 1),
                nn.BatchNorm1d(ff_size),
                nn.PReLU(ff_size),
                nn.Conv1d(ff_size, 1, 1)
            )
        else:
            raise TypeError('Unrecognized pool type {}'.format(pool_type))
        self.pool_type = pool_type

    def forward(self, x):
        # IMPORTANTLY: x must be composed of 2 channels!
        # channels = [clean/enhanced, noisy] in this order.
        # so its dim is (bsz, channels, time)
        # chunk 2 channels separately
        c_e, noisy = torch.chunk(x, 2, dim=1)
        # encode noisy
        noisy = self.frontend(noisy)
        hact = {'fe_noisy':noisy}
        if not self.ft_fe:
            noisy = noisy.detach()
        # pool the input c_e begin downsampling loop
        h = c_e
        for l_i, enc_layer in enumerate(self.enc_blocks):
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
            h = enc_layer(h)
            hact['enc_{}'.format(l_i)] = h
        # final pooling and classifier
        if self.pool_type == 'mha':
            n_W = self.W_noisy(noisy)
            ce_W = self.W_ce(h)
            # make attention maps
            h, att = self.mha(n_W.transpose(1, 2),
                              ce_W.transpose(1, 2),
                              ce_W.transpose(1, 2))
            h = h.transpose(1, 2)
            h = self.mha_norm(h)
            hact['att'] = att
            y = self.mlp(h)
        elif self.pool_type == 'mlp':
            h = torch.cat((h, noisy), dim=1)
            hact['att'] = None
            y = self.mlp(h.view(h.size(0), -1))
        else:
            # TODO: correct channels/time dimensions
            raise NotImplementedError
            h1, h2 = torch.chunk(h, 2, dim=0)
            hact['frontend_1'] = h1
            hact['frontend_2'] = h2
            h = torch.cat((h1, h2), dim=1)
            hact['att'] = None
            y_global = self.mlp(h.view(h.size(0), -1))
            y_local = self.mlp_l(h)
            hact['y_global'] = y_global
            hact['y_local'] = y_local
            y = torch.cat((y_global.view(-1, 1),
                           y_local.view(-1, 1)),
                          dim=0)
        return y, hact
        

class Discriminator(Model):
    
    def __init__(self, ninputs, fmaps,
                 kwidth, poolings,
                 pool_type='none',
                 pool_slen=None,
                 norm_type='bnorm',
                 bias=True,
                 phase_shift=None, 
                 sinc_conv=False,
                 num_spks=None):
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
    """
    disc = Discriminator(2, [64, 128, 256, 512, 1024],
                         31, [4] * 5, pool_type='none',
                         pool_slen=16)
    print(disc)
    print('Num params: ', disc.get_n_params())
    x = torch.randn(1, 2, 16384)
    y, _ = disc(x)
    print(y)
    print('x size: {} -> y size: {}'.format(x.size(), y.size()))
    """
    x = torch.randn(5, 2, 8000)
    fe = wf_builder('../../cfg/frontend_RF160ms_norm-emb100.cfg')
    D = DiscriminatorFE(frontend=fe, pool_type='mlp')
    y, h = D(x)
    print(D)
    print('y size: ', y.size())
    
