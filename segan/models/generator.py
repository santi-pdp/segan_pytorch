import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils as nnu
import torch.nn as nn
import random
import numpy as np
try:
    from core import *
    from attention import *
except ImportError:
    from .core import *
    from .attention import *

#if int(torch.__version__[2]) > 4:
from torch.nn.utils.spectral_norm import spectral_norm
#else:
#    from .spectral_norm import SpectralNorm as spectral_norm


class GSkip(nn.Module):

    def __init__(self, skip_type, size, skip_init, skip_dropout=0,
                 merge_mode='sum', cuda=False):
        # skip_init only applies to alpha skips
        super().__init__()
        self.merge_mode = merge_mode
        if skip_type == 'alpha' or skip_type == 'constant':
            if skip_init == 'zero':
                alpha_ = torch.zeros(size)
            elif skip_init == 'randn':
                alpha_ = torch.randn(size)
            elif skip_init == 'one':
                alpha_ = torch.ones(size)
            else:
                raise TypeError('Unrecognized alpha init scheme: ', 
                                skip_init)
            if cuda:
                alpha_ = alpha_.cuda()
            if skip_type == 'alpha':
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
            else:
                # constant, not learnable
                self.skip_k = Variable(alpha_, requires_grad=False)
                self.skip_k = self.skip_k.view(1, -1, 1)
        elif skip_type == 'conv':
            self.skip_k = nn.Conv1d(size, size, 11, stride=1,
                                    padding=11//2)
        else:
            raise TypeError('Unrecognized GSkip scheme: ', skip_type)
        self.skip_type = skip_type
        if skip_dropout > 0:
            self.skip_dropout = nn.Dropout(skip_dropout)

    def forward(self, hj, hi):
        if self.skip_type == 'conv':
            sk_h = self.skip_k(hj)
        else:
            skip_k = self.skip_k.repeat(hj.size(0), 1, hj.size(2))
            sk_h =  skip_k * hj
        if hasattr(self, 'skip_dropout'):
            sk_h = self.skip_dropout(sk_h)
        if self.merge_mode == 'sum':
            # merge with input hi on current layer
            return sk_h + hi
        elif self.merge_mode == 'concat':
            return torch.cat((hi, sk_h), dim=1)
        else:
            raise TypeError('Unrecognized skip merge mode: ', self.merge_mode)

class LinterpAffine(nn.Module):

    def __init__(self, num_params=1, std=1, bias=0):
        super().__init__()
        self.linterp_w = nn.Parameter(std * torch.randn(num_params))
        self.linterp_b = nn.Parameter(torch.ones(num_params) * bias)

    def forward(self, x):
        return self.linterp_w.view(1, -1, 1) * x + self.linterp_b.view(1, -1,
                                                                       1)


class GBlock(nn.Module):

    def __init__(self, ninputs, fmaps, kwidth,
                 activation, padding=None,
                 lnorm=False, dropout=0.,
                 pooling=2, enc=True, bias=False,
                 aal_h=None, linterp=False, snorm=False, 
                 convblock=False, satt=False, linterp_mode='linear',
                 comb=False):
        # linterp: do linear interpolation instead of simple conv transpose
        # snorm: spectral norm
        # comb: use comb filter block after deconv of same stride
        super().__init__()
        self.pooling = pooling
        self.linterp = linterp
        self.linterp_mode = linterp_mode
        self.enc = enc
        self.kwidth = kwidth
        self.convblock= convblock
        self.satt = satt
        if padding is None:
            padding = 0
        self.padding = padding
        if enc:
            if aal_h is not None:
                self.aal_conv = nn.Conv1d(ninputs, ninputs, 
                                          aal_h.shape[0],
                                          stride=1,
                                          padding=aal_h.shape[0] // 2 - 1,
                                          bias=False)
                if snorm:
                    self.aal_conv = spectral_norm(self.aal_conv)
                # apply AAL weights, reshaping impulse response to match
                # in channels and out channels
                aal_t = torch.FloatTensor(aal_h).view(1, 1, -1)
                aal_t = aal_t.repeat(ninputs, ninputs, 1)
                self.aal_conv.weight.data = aal_t
            if convblock:
                self.conv = Conv1DResBlock(ninputs, fmaps, kwidth,
                                           stride=pooling, bias=bias)
            else:
                self.conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                      stride=pooling,
                                      padding=padding,
                                      bias=bias)
            if snorm:
                self.conv = spectral_norm(self.conv)
            if activation == 'glu':
                # TODO: REVIEW
                raise NotImplementedError
                #self.glu_conv = nn.Conv1d(ninputs, fmaps, kwidth,
                #                          stride=pooling,
                #                          padding=padding,
                #                          bias=bias)
                if snorm:
                    self.glu_conv = spectral_norm(self.glu_conv)
        else:
            if linterp:
                self.linterp_aff = LinterpAffine(ninputs, std=0.1)
                self.conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                      stride=1, padding=kwidth//2,
                                      bias=bias)
                if snorm:
                    self.conv = spectral_norm(self.conv)
                if activation == 'glu':
                    raise NotImplementedError
                    #self.glu_conv = nn.Conv1d(ninputs, fmaps, kwidth,
                    #                          stride=1, padding=kwidth//2,
                    #                          bias=bias)
                    if snorm:
                        self.glu_conv = spectral_norm(self.glu_conv)
            else:
                if convblock:
                    self.conv = Conv1DResBlock(ninputs, fmaps, kwidth,
                                               stride=pooling, bias=bias, 
                                               transpose=True)
                else:
                    # decoder like with transposed conv
                    # compute padding required based on pooling
                    pad = (2 * pooling - pooling - kwidth)//-2
                    pad = max(pad, 0)
                    print('transpose with pad: {} kwidth:{} ninputs:{}'
                          ' fmaps:{}'.format(pad, kwidth, ninputs,
                                             fmaps))
                    
                    self.conv = nn.ConvTranspose1d(ninputs, fmaps, kwidth,
                                                   stride=pooling,
                                                   padding=pad,
                                                   output_padding=0,
                                                   bias=bias)
                if comb:
                    self.comb = CombFilter(fmaps, fmaps, pooling)
                if snorm:
                    self.conv = spectral_norm(self.conv)

                if activation == 'glu':
                    # TODO: REVIEW
                    raise NotImplementedError
                    #self.glu_conv = nn.ConvTranspose1d(ninputs, fmaps, kwidth,
                    #                                   stride=pooling,
                    #                                   padding=padding,
                    #                                   output_padding=pooling-1,
                    #                                   bias=bias)
                    if snorm:
                        self.glu_conv = spectral_norm(self.glu_conv)
        if satt:
            self.att = MultiHeadedAttention(1, fmaps, dropout=dropout)
        if activation is not None:
            self.act = activation
        if lnorm:
            self.ln = nn.InstanceNorm1d(fmaps, affine=True)
        if dropout > 0:
            self.dout = nn.Dropout(dropout)

    def forward(self, x, att_weight=0.):
        if len(x.size()) == 4:
            # inverse case from 1D -> 2D, go 2D -> 1D
            # re-format input from [B, K, C, L] to [B, K * C, L]
            # where C: frequency, L: time
            x = x.squeeze(1)
        if hasattr(self, 'aal_conv'):
            x = self.aal_conv(x)
        if self.linterp:
            #x = self.pre_conv(x)
            x = F.upsample(x, scale_factor=self.pooling,
                           mode=self.linterp_mode)
            x = self.linterp_aff(x)
        if self.enc and self.padding == 0:
            if self.pooling == 1:
                # apply proper padding
                x = F.pad(x, ((self.kwidth//2), self.kwidth//2))
            else:
                # apply proper padding
                x = F.pad(x, ((self.kwidth//2)-1, self.kwidth//2))

        h = self.conv(x)
        if not self.enc and not self.linterp and not self.convblock \
            and self.kwidth % 2 != 0:
            # trim last value of h perque el kernel es imparell
            # TODO: generalitzar a kernel parell/imparell
            #print('h size: ', h.size())
            h = h[:, :, :-1]
        if hasattr(self, 'comb'):
            h = self.comb(h)
        linear_h = h
        if hasattr(self, 'act'):
            if self.act == 'glu':
                hg = self.glu_conv(x)
                h = h * F.sigmoid(hg)
            else:
                h = self.act(h)
        if hasattr(self, 'ln'):
            h = self.ln(h)
        if hasattr(self, 'att'):
            #print('Applying self-attention in GBlock')
            #print('Satt h input size: ', h.size())
            if att_weight > 0:
                h_t = h.transpose(1, 2)
                o = self.att(h_t, h_t, h_t)
                #print('satt o size: ', o.size())
                o = o.transpose(1, 2)
                h = h + att_weight * o
        if hasattr(self, 'dout'):
            h = self.dout(h)
        return h, linear_h


class G2Block(nn.Module):
    """ Conv2D Generator Blocks """

    def __init__(self, ninputs, fmaps, kwidth,
                 activation, padding=None,
                 bnorm=False, dropout=0.,
                 pooling=2, enc=True, bias=False):
        super().__init__()
        if padding is None:
            padding = (kwidth // 2)
        if enc:
            self.conv = nn.Conv2d(ninputs, fmaps, kwidth,
                                  stride=pooling,
                                  padding=padding,
                                  bias=bias)
        else:
            # decoder like with transposed conv
            self.conv = nn.ConvTranspose2d(ninputs, fmaps, kwidth,
                                           stride=pooling,
                                           padding=padding)
        if bnorm:
            self.bn = nn.BatchNorm2d(fmaps)
        if activation is not None:
            self.act = activation
        if dropout > 0:
            self.dout = nn.Dropout2d(dropout)

    def forward(self, x):
        if len(x.size()) == 3:
            # re-format input from [B, C, L] to [B, 1, C, L]
            # where C: frequency, L: time
            x = x.unsqueeze(1)
        h = self.conv(x)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'act'):
            h = self.act(h)
        if hasattr(self, 'dout'):
            h = self.dout(h)
        return h

class CombFilter(nn.Module):

    def __init__(self, ninputs, fmaps, L):
        super().__init__()
        self.L = L
        self.filt = nn.Conv1d(ninputs, fmaps, 2, dilation=L, bias=False)
        r_init_weight = torch.ones(ninputs * fmaps, 2)
        r_init_weight[:, 0] = torch.rand(r_init_weight.size(0))
        self.filt.weight.data = r_init_weight.view(fmaps, ninputs, 2)

    def forward(self, x):
        x_p = F.pad(x, (self.L, 0))
        y = self.filt(x_p)
        return y

class PostProcessingCombNet(nn.Module):

    def __init__(self, ninputs, fmaps, L=[4, 8, 16, 32]):
        super().__init__()
        filts = nn.ModuleList()
        for l in L:
            filt = CombFilter(ninputs, fmaps//len(L), l)
            filts.append(filt)
        self.filts = filts
        self.W = nn.Linear(fmaps, 1, bias=False)

    def forward(self, x):
        hs = []
        for filt in self.filts:
            h = filt(x)
            hs.append(h)
            #print('Comb h: ', h.size())
        hs = torch.cat(hs, dim=1)
        #print('hs size: ', hs.size())
        y = self.W(hs.transpose(1, 2)).transpose(1, 2)
        return y

class OutGate(nn.Module):

    def __init__(self, ninputs, noutputs):
        super().__init__()
        self.G = nn.Conv1d(ninputs, noutputs, 361, padding=361//2)
        self.W = nn.Conv1d(ninputs, noutputs, 361, padding=361//2)

    def forward(self, x):
        return F.sigmoid(self.G(x)) * F.tanh(self.W(x))


class Generator1D(Model):

    def __init__(self, ninputs, enc_fmaps, kwidth,
                 activations, lnorm=False, dropout=0.,
                 pooling=2, z_dim=256, z_all=False,
                 skip=True, skip_blacklist=[],
                 dec_activations=None, cuda=False,
                 bias=False, aal=False, wd=0.,
                 skip_init='one', skip_dropout=0.,
                 no_tanh=False, aal_out=False,
                 rnn_core=False, linterp=False,
                 mlpconv=False, dec_kwidth=None,
                 subtract_mean=False, no_z=False,
                 skip_type='alpha', 
                 num_spks=None, multilayer_out=False,
                 skip_merge='sum', snorm=False,
                 convblock=False, post_skip=False,
                 pos_code=False, satt=False,
                 dec_fmaps=None, up_poolings=None,
                 post_proc=False, out_gate=False, 
                 linterp_mode='linear', hidden_comb=False, 
                 big_out_filter=False, z_std=1):
        # if num_spks is specified, do onehot coditioners in dec stages
        # subract_mean: from output signal, get rif of mean by windows
        # multilayer_out: add some convs in between gblocks in decoder
        super().__init__(name='Generator1D')
        self.dec_kwidth = dec_kwidth
        self.skip = skip
        self.skip_init = skip_init
        self.skip_dropout = skip_dropout
        self.subtract_mean = subtract_mean
        self.snorm = snorm
        self.z_dim = z_dim
        self.z_all = z_all
        self.pos_code = pos_code
        self.post_skip = post_skip
        self.big_out_filter = big_out_filter
        self.satt = satt
        self.post_proc = post_proc
        self.onehot = num_spks is not None
        if self.onehot:
            assert num_spks > 0
        self.num_spks = num_spks
        # do not place any z
        self.no_z = no_z
        self.do_cuda = cuda
        self.wd = wd
        self.no_tanh = no_tanh
        self.skip_blacklist = skip_blacklist
        self.z_std = z_std
        self.gen_enc = nn.ModuleList()
        if aal or aal_out:
            # Make cheby1 filter to include into pytorch conv blocks
            from scipy.signal import cheby1, dlti, dimpulse
            system = dlti(*cheby1(8, 0.05, 0.8 / pooling))
            tout, yout = dimpulse(system)
            filter_h = yout[0]
        if aal:
            self.filter_h = filter_h
        else:
            self.filter_h = None

        if dec_kwidth is None:
            dec_kwidth = kwidth

        if isinstance(activations, str):
            if activations != 'glu':
                activations = getattr(nn, activations)()
        if not isinstance(activations, list):
            activations = [activations] * len(enc_fmaps)
        if not isinstance(pooling, list) or len(pooling) == 1: 
            pooling = [pooling] * len(enc_fmaps)
        skips = {}
        # Build Encoder
        for layer_idx, (fmaps, pool, act) in enumerate(zip(enc_fmaps, 
                                                           pooling,
                                                           activations)):
            if layer_idx == 0:
                inp = ninputs
            else:
                inp = enc_fmaps[layer_idx - 1]
            if self.skip and layer_idx < (len(enc_fmaps) - 1):
                if layer_idx not in self.skip_blacklist:
                    l_i = layer_idx
                    gskip = GSkip(skip_type, fmaps,
                                  skip_init,
                                  skip_dropout,
                                  merge_mode=skip_merge,
                                  cuda=self.do_cuda)
                    skips[l_i] = {'alpha':gskip}
                    setattr(self, 'alpha_{}'.format(l_i), skips[l_i]['alpha'])
            self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                       padding=None, lnorm=lnorm, 
                                       dropout=dropout, pooling=pool,
                                       enc=True, bias=bias, 
                                       aal_h=self.filter_h,
                                       snorm=snorm, convblock=convblock,
                                       satt=self.satt))
        self.skips = skips
        dec_inp = enc_fmaps[-1]
        if dec_fmaps is None:
            if mlpconv:
                dec_fmaps = enc_fmaps[:-1][::-1] + [16, 8, 1]
                print(dec_fmaps)
                up_poolings = [pooling] * (len(dec_fmaps) - 2) + [1] * 3
                add_activations = [nn.PReLU(16), nn.PReLU(8), nn.PReLU(1)]
                raise NotImplementedError('MLPconv is not useful and should be'
                                          ' deleted')
            else:
                dec_fmaps = enc_fmaps[:-1][::-1] + [1]
                up_poolings = pooling[::-1]
                #up_poolings = [pooling] * len(dec_fmaps)
            print('up_poolings: ', up_poolings)
            self.up_poolings = up_poolings
        else:
            assert up_poolings is not None
            self.up_poolings = up_poolings
        if rnn_core:
            self.z_all = False
            z_all = False
            # place a bidirectional RNN layer in the core to condition
            # everything to everything AND Z will be the init state of it
            self.rnn_core = nn.LSTM(dec_inp, dec_inp // 2, bidirectional=True,
                                    batch_first=True)
        else:
            if no_z:
                all_z = False
            else:
                dec_inp += z_dim
        #print(dec_fmaps)
        # Build Decoder
        self.gen_dec = nn.ModuleList()

        if dec_activations is None:
            # assign same activations as in Encoder
            dec_activations = [activations[0]] * len(dec_fmaps)
        else:
            if mlpconv:
                dec_activations = dec_activations[:-1]
                dec_activations += add_activations
        
        enc_layer_idx = len(enc_fmaps) - 1
        for layer_idx, (fmaps, act) in enumerate(zip(dec_fmaps, 
                                                     dec_activations)):
            if skip and layer_idx > 0 and enc_layer_idx not in skip_blacklist \
                and up_poolings[layer_idx] > 1: 
                if skip_merge == 'concat':
                    dec_inp *= 2
                print('Added skip conn input of enc idx: {} and size:'
                      ' {}'.format(enc_layer_idx, dec_inp))

            if z_all and layer_idx > 0:
                dec_inp += z_dim

            if self.onehot:
                dec_inp += self.num_spks

            if layer_idx >= len(dec_fmaps) - 1:
                if self.no_tanh:
                    act = None
                else:
                    act = nn.Tanh()
                lnorm = False
                dropout = 0
            if up_poolings[layer_idx] > 1:
                pooling = up_poolings[layer_idx]
                self.gen_dec.append(GBlock(dec_inp,
                                           fmaps, dec_kwidth, act, 
                                           padding=0, 
                                           lnorm=lnorm,
                                           dropout=dropout, pooling=pooling, 
                                           enc=False,
                                           bias=bias,
                                           linterp=linterp, 
                                           linterp_mode=linterp_mode,
                                           convblock=convblock, 
                                           comb=hidden_comb))
            else:
                self.gen_dec.append(GBlock(dec_inp,
                                           fmaps, dec_kwidth, act, 
                                           lnorm=lnorm,
                                           dropout=dropout, pooling=1,
                                           padding=kwidth//2,
                                           enc=True,
                                           bias=bias,
                                           convblock=convblock))
            dec_inp = fmaps
        if aal_out:
            # make AAL filter to put in output
            self.aal_out = nn.Conv1d(1, 1, filter_h.shape[0] + 1,
                                     stride=1, 
                                     padding=filter_h.shape[0] // 2,
                                     bias=False)
            print('filter_h shape: ', filter_h.shape)
            # apply AAL weights, reshaping impulse response to match
            # in channels and out channels
            aal_t = torch.FloatTensor(filter_h).view(1, 1, -1)
            aal_t = torch.cat((aal_t, torch.zeros(1, 1, 1)), dim=-1)
            self.aal_out.weight.data = aal_t
            print('aal_t size: ', aal_t.size())

        if post_proc:
            self.comb_net = PostProcessingCombNet(1, 512)
        if out_gate:
            self.out_gate = OutGate(1, 1)
        if big_out_filter:
            self.out_filter = nn.Conv1d(1, 1, 513, padding=513//2)

        

    def forward(self, x, z=None, ret_hid=False, spkid=None, 
                slice_idx=0, att_weight=0):
        if self.num_spks is not None and spkid is None:
            raise ValueError('Please specify spk ID to network to '
                             'build OH identifier in decoder')

        hall = {}
        hi = x
        skips = self.skips
        for l_i, enc_layer in enumerate(self.gen_enc):
            hi, linear_hi = enc_layer(hi, att_weight=att_weight)
            #print('ENC {} hi size: {}'.format(l_i, hi.size()))
                    #print('Adding skip[{}]={}, alpha={}'.format(l_i,
                    #                                            hi.size(),
                    #                                            hi.size(1)))
            if self.skip and l_i < (len(self.gen_enc) - 1):
                if l_i not in self.skip_blacklist:
                    if self.post_skip:
                        skips[l_i]['tensor'] = hi
                    else:
                        skips[l_i]['tensor'] = linear_hi
            if ret_hid:
                hall['enc_{}'.format(l_i)] = hi
        if hasattr(self, 'rnn_core'):
            self.z_all = False
            if z is None:
                # make z as initial RNN state forward and backward
                # (2 directions)
                if self.no_z:
                    # MAKE DETERMINISTIC ZERO
                    h0 = Variable(torch.zeros(2, hi.size(0), hi.size(1)//2))
                else:
                    h0 = Variable(self.z_std * torch.randn(2, 
                                                           hi.size(0), 
                                                           hi.size(1)//2))
                c0 = Variable(torch.zeros(2, hi.size(0), hi.size(1)//2))
                if self.do_cuda:
                    h0 = h0.cuda()
                    c0 = c0.cuda()
                z = (h0, c0)
                if not hasattr(self, 'z'):
                    self.z = z
            # Conv --> RNN
            hi = hi.transpose(1, 2)
            hi, state = self.rnn_core(hi, z)
            # RNN --> Conv
            hi = hi.transpose(1, 2)
        else:
            if not self.no_z:
                if z is None:
                    # make z 
                    z = Variable(self.z_std * torch.randn(hi.size(0), self.z_dim,
                                                         *hi.size()[2:]))
                    #print('Made z of dim: ', z.size())
                if len(z.size()) != len(hi.size()):
                    raise ValueError('len(z.size) {} != len(hi.size) {}'
                                     ''.format(len(z.size()), len(hi.size())))
                if self.do_cuda:
                    z = z.cuda()
                if not hasattr(self, 'z'):
                    self.z = z
                #print('Concating z {} and hi {}'.format(z.size(),
                #                                        hi.size()))
                hi = torch.cat((z, hi), dim=1)
                if ret_hid:
                    hall['enc_zc'] = hi
            else:
                z = None
            if self.pos_code:
                hi = pos_code(slice_idx, hi)
        #print('Concated hi|z size: ', hi.size())
        enc_layer_idx = len(self.gen_enc) - 1
        z_up = z
        if self.onehot:
            # make one hot identifier batch
            spk_oh = Variable(torch.zeros(spkid.size(0), 
                                          self.num_spks))
            for bidx in range(spkid.size(0)):
                if len(spkid.size()) == 3:
                    spk_id = spkid[bidx, 0].cpu().data[0]
                else:
                    spk_id = spkid[bidx].cpu().data[0]
                spk_oh[bidx, spk_id] = 1
            spk_oh = spk_oh.view(spk_oh.size(0), -1, 1)
            if self.do_cuda:
                spk_oh = spk_oh.cuda()
            # Now one-hot is [B, SPKS, 1] ready to be 
            # repeated to [B, SPKS, T] depending on layer
        for l_i, dec_layer in enumerate(self.gen_dec):
            if self.skip and enc_layer_idx in self.skips and \
            self.up_poolings[l_i] > 1:
                skip_conn = skips[enc_layer_idx]
                #hi = self.skip_merge(skip_conn, hi)
                #print('Merging  hi {} with skip {} of hj {}'.format(hi.size(),
                #                                                    l_i,
                #                                                    skip_conn['tensor'].size()))
                hi = skip_conn['alpha'](skip_conn['tensor'], hi)
            if l_i > 0 and self.z_all:
                # concat z in every layer
                z_up = torch.cat((z_up, z_up), dim=2)
                hi = torch.cat((hi, z_up), dim=1)
            if self.onehot:
                # repeat one-hot in time to adjust to concat
                spk_oh_r = spk_oh.repeat(1, 1, hi.size(-1))
                # concat in depth (channels)
                hi = torch.cat((hi, spk_oh_r), dim=1)
            #print('DEC in size after skip and z_all: ', hi.size())
            #print('decoding layer {} with input {}'.format(l_i, hi.size()))
            hi, _ = dec_layer(hi, att_weight=att_weight)
            #print('decoding layer {} output {}'.format(l_i, hi.size()))
            enc_layer_idx -= 1
            if ret_hid:
                hall['dec_{}'.format(l_i)] = hi
        if hasattr(self, 'aal_out'):
            hi = self.aal_out(hi)
        if self.subtract_mean:
            hi = self.subtract_windowed_wav_mean(hi)
        if hasattr(self, 'comb_net'):
            hi = F.tanh(self.comb_net(hi))
        if hasattr(self, 'out_gate'):
            hi = self.out_gate(hi)
        if hasattr(self, 'out_filter'):
            hi = self.out_filter(hi)
        # normalize G output in range within [-1, 1]
        #hi = self.batch_minmax_norm(hi)
        if ret_hid:
            return hi, hall
        else:
            return hi

    def batch_minmax_norm(self, x, out_min=-1, out_max=1):
        mins = torch.min(x, dim=2)[0]
        maxs = torch.max(x, dim=2)[0]
        R = (out_max - out_min) / (maxs - mins)
        R = R.unsqueeze(1)
        #print('R size: ', R.size())
        #print('x size: ', x.size())
        #print('mins size: ', mins.size())
        x = R * (x - mins.unsqueeze(1)) + out_min
        #print('norm x size: ', x.size())
        return x

    def subtract_windowed_wav_mean(self, wavb, W=20):
        cwavb = Variable(torch.zeros(wavb.size()))
        if self.do_cuda:
            cwavb = cwavb.cuda()
        for n in range(0, wavb.size(2), W):
            mn = torch.mean(wavb[:, :, n:n + W])
            cwavb[:, :, n:n + W] = wavb[:, :, n:n + W] - mn
        return cwavb

    def skip_merge(self, skip_conn, hi):
        # TODO: DEPRECATED WITH NEW SKIP SCHEME
        raise NotImplementedError
        hj = skip_conn['tensor']
        alpha = skip_conn['alpha'].view(1, -1, 1)
        alpha = alpha.repeat(hj.size(0), 1, hj.size(2))
        #print('hi: ', hi.size())
        #print('hj: ', hj.size())
        #print('alpha: ', alpha.size())
        #print('alpha: ', alpha)
        if 'dropout' in skip_conn:
            alpha = skip_conn['dropout'](alpha)
            #print('alpha: ', alpha)
        return hi + alpha * hj
        

class Generator(Model):

    def __init__(self, ninputs, enc_fmaps, kwidth, 
                 activations, bnorm=False, dropout=0.,
                 pooling=2, z_dim=1024, z_all=False,
                 skip=True, skip_blacklist=[],
                 dec_activations=None, cuda=False,
                 bias=False, aal=False, wd=0.,
                 core2d=False, core2d_kwidth=None, 
                 core2d_felayers=1,
                 skip_mode='concat'):
        # aal: anti-aliasing filter prior to each striding conv in enc
        super().__init__(name='Generator')
        self.skip_mode = skip_mode
        self.skip = skip
        self.z_dim = z_dim
        self.z_all = z_all
        self.do_cuda = cuda
        self.core2d = core2d
        self.wd = wd
        self.skip_blacklist = skip_blacklist
        if core2d_kwidth is None:
            core2d_kwidth = kwidth
        self.gen_enc = nn.ModuleList()
        if aal:
            # Make cheby1 filter to include into pytorch conv blocks
            from scipy.signal import cheby1, dlti, dimpulse
            system = dlti(*cheby1(8, 0.05, 0.8 / 2))
            tout, yout = dimpulse(system)
            filter_h = yout[0]
            self.filter_h = filter_h
        else:
            self.filter_h = None

        if isinstance(activations, str):
            activations = getattr(nn, activations)()
        if not isinstance(activations, list):
            activations = [activations] * len(enc_fmaps)
        # always begin with 1D block
        for layer_idx, (fmaps, act) in enumerate(zip(enc_fmaps, 
                                                     activations)):
            if layer_idx == 0:
                inp = ninputs
            else:
                inp = enc_fmaps[layer_idx - 1]
            if core2d:
                if layer_idx < core2d_felayers:
                    self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                               padding=None, bnorm=bnorm, 
                                               dropout=dropout, pooling=pooling,
                                               enc=True, bias=bias, 
                                               aal_h=self.filter_h,
                                               att_weight=att_weight))
                else:
                    if layer_idx == core2d_felayers:
                        # fmaps is 1 after conv1d blocks
                        inp = 1
                    self.gen_enc.append(G2Block(inp, fmaps, core2d_kwidth, act,
                                                padding=None, bnorm=bnorm, 
                                                dropout=dropout, pooling=pooling,
                                                enc=True, bias=bias))
            else:
                self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                           padding=None, bnorm=bnorm, 
                                           dropout=dropout, pooling=pooling,
                                           enc=True, bias=bias, 
                                           aal_h=self.filter_h,
                                           att_weight=att_weight))
        dec_inp = enc_fmaps[-1]
        if self.core2d:
            #dec_fmaps = enc_fmaps[::-1][1:-2]+ [1, 1]
            dec_fmaps = enc_fmaps[::-1][:-2] + [1, 1]
        else:
            dec_fmaps = enc_fmaps[::-1][1:]+ [1] 
        #print(dec_fmaps)
        #print(enc_fmaps)
        #print('dec_fmaps: ', dec_fmaps)
        self.gen_dec = nn.ModuleList()
        if dec_activations is None:
            dec_activations = activations
        
        dec_inp += z_dim

        for layer_idx, (fmaps, act) in enumerate(zip(dec_fmaps, 
                                                     dec_activations)):
            if skip and layer_idx > 0 and layer_idx not in skip_blacklist:
                #print('Adding skip conn input of idx: {} and size:'
                #      ' {}'.format(layer_idx, dec_inp))
                if self.skip_mode == 'concat':
                    dec_inp += enc_fmaps[-(layer_idx+1)]

            if z_all and layer_idx > 0:
                dec_inp += z_dim

            if layer_idx >= len(dec_fmaps) - 1:
                #act = None #nn.Tanh()
                act = nn.Tanh()
                bnorm = False
                dropout = 0

            if layer_idx < len(dec_fmaps) -1 and core2d:
                self.gen_dec.append(G2Block(dec_inp,
                                            fmaps, core2d_kwidth + 1, act, 
                                            padding=core2d_kwidth//2, 
                                            bnorm=bnorm,
                                            dropout=dropout, pooling=pooling, 
                                            enc=False,
                                            bias=bias))
            else:
                if layer_idx == len(dec_fmaps) - 1:
                    # after conv2d channel condensation, fmaps mirror the ones
                    # extracted in 1D encoder
                    dec_inp = enc_fmaps[0]
                    if skip and layer_idx not in skip_blacklist:
                        dec_inp += enc_fmaps[-(layer_idx+1)]
                self.gen_dec.append(GBlock(dec_inp,
                                           fmaps, kwidth + 1, act, 
                                           padding=kwidth//2, 
                                           bnorm=bnorm,
                                           dropout=dropout, pooling=pooling, 
                                           enc=False,
                                           bias=bias))
            dec_inp = fmaps

    def forward(self, x, z=None):
        hi = x
        skips = []
        for l_i, enc_layer in enumerate(self.gen_enc):
            hi = enc_layer(hi)
            #print('ENC {} hi size: {}'.format(l_i, hi.size()))
            if self.skip and l_i < (len(self.gen_enc) - 1):
                #print('Appending skip connection')
                skips.append(hi)
            #print('hi size: ', hi.size())
        #print('=' * 50)
        skips = skips[::-1]
        if z is None:
            # make z 
            #z = Variable(torch.randn(x.size(0), self.z_dim, hi.size(2)))
            #z = Variable(torch.randn(*hi.size()))
            z = Variable(torch.randn(hi.size(0), self.z_dim,
                                     *hi.size()[2:]))
        if len(z.size()) != len(hi.size()):
            raise ValueError('len(z.size) {} != len(hi.size) {}'
                             ''.format(len(z.size()), len(hi.size())))
        if self.do_cuda:
            z = z.cuda()
        if not hasattr(self, 'z'):
            self.z = z
        #print('z size: ', z.size())
        hi = torch.cat((hi, z), dim=1)
        #print('Input to dec after concating z and enc out: ', hi.size())
        #print('Enc out size: ', hi.size())
        z_up = z
        for l_i, dec_layer in enumerate(self.gen_dec):
            #print('dec layer: {} with input: {}'.format(l_i, hi.size()))
            #print('DEC in size: ', hi.size())
            if self.skip and l_i > 0 and l_i not in self.skip_blacklist:
                skip_conn = skips[l_i - 1]
                #print('concating skip {} to hi {}'.format(skip_conn.size(),
                #                                          hi.size()))
                hi = self.skip_merge(skip_conn, hi)
                #print('Merged hi: ', hi.size())
                #hi = torch.cat((hi, skip_conn), dim=1)
            if l_i > 0 and self.z_all:
                # concat z in every layer
                #print('z.size: ', z.size())
                z_up = torch.cat((z_up, z_up), dim=2)
                hi = torch.cat((hi, z_up), dim=1)
            hi = dec_layer(hi)
            #print('-' * 20)
            #print('hi size: ', hi.size())
        return hi

    def skip_merge(self, skip, hi):
        if self.skip_mode == 'concat':
            if len(hi.size()) == 4 and len(skip.size()) == 3:
                hi = hi.squeeze(1)
            # 1-D case
            hi_ = torch.cat((skip, hi), dim=1)
        elif self.skip_mode == 'sum':
            hi_ = skip + hi
        else:
            raise ValueError('Urecognized skip mode: ', self.skip_mode)
        return hi_


    def parameters(self):
        params = []
        for k, v in self.named_parameters():
            if 'aal_conv' not in k:
                params.append({'params':v, 'weight_decay':self.wd})
            else:
                print('Excluding param: {} from Genc block'.format(k))
        return params


class AttGenerator1D(Model):

    def __init__(self, ninputs, enc_fmaps, kwidth,
                 dec_kwidth, pooling=2, 
                 cuda=False, skip=True, skip_merge='concat',
                 snorm=False):
        super().__init__(name='AttGenerator1D')
        self.ninputs = ninputs
        self.enc_fmaps = enc_fmaps
        self.kwidth = kwidth
        self.dec_kwidth = dec_kwidth
        self.pooling = pooling
        self.do_cuda = cuda
        self.snorm = snorm
        self.skip = skip
        skips = {}
        activations = [nn.PReLU(fmaps) for fmaps in enc_fmaps]
        self.gen_enc = nn.ModuleList()
        # Build Encoder
        for layer_idx, (fmaps, act) in enumerate(zip(enc_fmaps, 
                                                     activations)):
            if layer_idx == 0:
                inp = ninputs
            else:
                inp = enc_fmaps[layer_idx - 1]
            if self.skip and layer_idx < (len(enc_fmaps) - 1):
                l_i = layer_idx
                gskip = GSkip('constant', fmaps,
                              'one', 0,
                              merge_mode='concat',
                              cuda=self.do_cuda)
                pad = (2 * pooling - pooling - 3)//-2
                #deconv = nn.Conv2d(1, 1, 3, 
                #                   stride=1,
                #                   padding=1)
                skips[l_i] = {'alpha':gskip}
                #              'deconv':deconv}
                setattr(self, 'alpha_{}'.format(l_i), skips[l_i]['alpha'])
                #setattr(self, 'deconv_{}'.format(l_i), skips[l_i]['deconv'])
            self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                       padding=None, lnorm=False,
                                       dropout=0, pooling=pooling,
                                       enc=True, bias=True,
                                       aal_h=None, snorm=snorm))
        # store skip connections and proceed to decoder
        self.skips = skips
        # Now decoder input size matches RNN size, which is in turn
        # z dimension in latent core
        rnn_inp = enc_fmaps[-1] * 2
        dec_inp = rnn_inp // 2
        self.z_dim = dec_inp
        # build dec fmaps
        dec_fmaps = enc_fmaps[:-1][::-1] + [1]
        up_poolings = [pooling] * len(dec_fmaps)
        print('up_poolings: ', up_poolings)
        self.up_poolings = up_poolings
        # build the attentive RNN
        self.rnn_att = nn.LSTM(rnn_inp // 2, rnn_inp // 2,
                               bidirectional=False,
                               batch_first=True)
        if snorm:
            spectral_norm(self.rnn_att, name='weight_hh_l0')
            spectral_norm(self.rnn_att, name='weight_ih_l0')
        self.attn = Attn(enc_fmaps[-1], cuda=self.do_cuda, snorm=snorm)
        # Build Decoder
        self.gen_dec = nn.ModuleList()
        dec_activations = [nn.PReLU(fmaps) for fmaps in dec_fmaps]
        enc_layer_idx = len(enc_fmaps) - 1
        for layer_idx, (fmaps, act) in enumerate(zip(dec_fmaps, 
                                                     dec_activations)):
            if skip and layer_idx > 0:
                if skip_merge == 'concat':
                    dec_inp *= 2
                print('Added skip conn input of enc idx: {} and size:'
                      ' {}'.format(enc_layer_idx, dec_inp))

            if layer_idx >= len(dec_fmaps) - 1:
                act = nn.Tanh()
            if up_poolings[layer_idx] > 1:
                self.gen_dec.append(GBlock(dec_inp,
                                           fmaps, dec_kwidth, act, 
                                           padding=0, 
                                           lnorm=False,
                                           dropout=0, pooling=pooling, 
                                           enc=False,
                                           bias=True,
                                           linterp=False,
                                           snorm=snorm))
            dec_inp = fmaps


    def forward(self, x, dec_steps, z=None, ret_hid=False, spkid=None):
        hall = {}
        hi = x
        skips = self.skips
        for l_i, enc_layer in enumerate(self.gen_enc):
            hi, linear_hi = enc_layer(hi)
            #print('ENC {} hi size: {}'.format(l_i, hi.size()))
            #print('Adding skip[{}]={}, alpha={}'.format(l_i,
            #                                            hi.size(),
            #                                            hi.size(1)))
            if self.skip and l_i < (len(self.gen_enc) - 1):
                skips[l_i]['tensor'] = linear_hi
            if ret_hid:
                hall['enc_{}'.format(l_i)] = hi
        # Now we have all encoder states, we need to forward through attn
        # recurrent net, initializing it with latent code
        if z is None:
            # make z 
            z = torch.randn(1, hi.size(0), self.z_dim)
            if self.do_cuda:
                z = z.cuda()
        if not hasattr(self, 'z'):
            # assign z to class if does not exist 
            self.z = z
        # build beg of seq token
        bos = torch.zeros(hi.size(0), 1, hi.size(1), requires_grad=True)
        if self.do_cuda:
            bos = bos.cuda()
        # transpose conv enc output to match rnn axis
        hi = hi.transpose(1, 2)
        prev_y = bos
        state = (F.tanh(z), z)
        hts = []
        att_map = []
        # Run RNN step by step to decode with attention in hand
        for dec_t in range(dec_steps):
            #print('--------- TSTEP {} -------------'.format(dec_t))
            attn_weights = self.attn(prev_y, hi)
            #print('attn_weights: ', attn_weights)
            #print('attn_weights size: ', attn_weights.size())
            #print('hi size: ', hi.size())
            c_vec = torch.bmm(attn_weights, hi)
            #print('c_vec size: ', c_vec.size())
            rnn_in = c_vec
            #print('rnn_in size: ', rnn_in.size())
            ht, state = self.rnn_att(rnn_in, state)
            prev_y = ht
            hts.append(ht)
            att_map.append(attn_weights)
            #print('ht size: ', ht.size())
        hts = torch.cat(hts, dim=1)
        #print('Final HTS size: ', hts.size())
        att_map = torch.cat(att_map, dim=1)
        #print('att_map size: ', att_map.size())
        hall['att'] = att_map
        hi = hts.transpose(1,2)
        # store interpolated attentions (through 2D deconv)
        int_atts = []
        prev_attn = att_map
        # ============================================
        enc_layer_idx = len(self.gen_enc) - 1
        trim_up = False
        curr_attn = None
        for l_i, dec_layer in enumerate(self.gen_dec):
            if self.skip and enc_layer_idx in self.skips and \
            self.up_poolings[l_i] > 1:
                skip_conn = skips[enc_layer_idx]
                prev_tensor = skip_conn['tensor']
                # DUMB SKIP CONNECTION: average each source tensor
                # in source axis direction without attention
                # interpolation
                # ------------------
                hj = torch.mean(prev_tensor, dim=2, keepdim=True)
                hj = hj.repeat(1, 1, hi.size(2))
                # -----------------
                """
                # we need to achieve same tsteps like prev_tensor
                # when interpolating attention
                #print('prev_tensor size: ', prev_tensor.size())
                if prev_tensor.size(2) > prev_attn.size(2) * 2:
                    #print('WARNING: SIZE WONT MATCH')
                    prev_attn = F.pad(prev_attn, (0, 1))
                    trim_up = True
                #print('prev_attn size: ', prev_attn.size())
                #print('prev attn: ', prev_attn)
                #print('prev attn max: ', prev_attn.max())
                prev_attn = prev_attn.unsqueeze(1)
                # linear interpolation and then conv with softmax
                new_attn = F.upsample(prev_attn, scale_factor=self.pooling,
                                      mode='bilinear', align_corners=False)
                if trim_up:
                    new_attn = new_attn[:, :, :, :-1]
                    #print('new_attn size: ', new_attn.size())
                    trim_up = False

                #print('new attn prior conv min: ', new_attn.min())
                #print('new attn prior conv  max: ', new_attn.max())
                new_attn = skip_conn['deconv'](new_attn)
                #print('new attn min: ', new_attn.min())
                #print('new attn max: ', new_attn.max())
                #print('new_attn size after squeeze: ', new_attn.size())
                int_atts.append(new_attn.squeeze(1))
                curr_attn = int_atts[-1]
                prev_attn = curr_attn
                #print('New attn size at dec layer {}:{}'.format(l_i,
                #                                                curr_attn.size()))
                curr_attn = curr_attn.transpose(1, 2)
                #print('curr_attn size after trans: ', curr_attn.size())
                #print('hi size: ', hi.size())
                #print('dot b/w {}x{}'.format(skip_conn['tensor'].size(),
                #                             curr_attn.size()))
                hj = torch.bmm(skip_conn['tensor'], curr_attn)
                #print('hj min: ', hj.min())
                #print('hj max: ', hj.max())
                #print('hj size: ', hj.size())
                #print('hi size: ', hi.size())
                # First, deconv attention at this level
                #hi = self.skip_merge(skip_conn, hi)
                #print('Merging  hi {} with skip {} of hj {}'.format(hi.size(),
                #                                                    l_i,
                #                                                    skip_conn['tensor'].size()))
                """
                hi = skip_conn['alpha'](hj, hi)
                if ret_hid and curr_attn is not None:
                    hall['att_{}'.format(l_i)] = curr_attn
            hi, _ = dec_layer(hi)
            #print('decoding layer {} output {}'.format(l_i, hi.size()))
            enc_layer_idx -= 1
            if ret_hid:
                hall['dec_{}'.format(l_i)] = hi.transpose(1, 2)
        #print('hi size: ', hi.size())
        if ret_hid:
            return hi, hall
        else:
            return hi

class NIGenerator1D(Model):
    """ Non Interpolated Generator 
        Avoiding deconvolutions or any upsampling method
        to work, after encoding, in the same time-scale
        as output. For this we generate a z vector
        with T samples, and inject decimated/interp input
        compression into these T samples:
        [z1|c1, z2|c2, ..., zT|cT]
    """
    def __init__(self, ninputs, enc_fmaps, kwidth,
                 activations, lnorm=False, dropout=0.,
                 pooling=2, z_dim=256, z_all=False,
                 skip=True, skip_blacklist=[],
                 dec_activations=None, cuda=False,
                 bias=False, aal=False, wd=0.,
                 skip_init='one', skip_dropout=0.,
                 no_tanh=False, aal_out=False,
                 rnn_core=False, linterp=False,
                 mlpconv=False, dec_kwidth=None,
                 subtract_mean=False, no_z=False,
                 skip_type='alpha', 
                 num_spks=None, multilayer_out=False,
                 skip_merge='sum', snorm=False,
                 convblock=False, post_skip=False,
                 pos_code=False, satt=False,
                 dec_fmaps=None, up_poolings=None,
                 post_proc=False, out_gate=False, 
                 linterp_mode='linear', hidden_comb=False):
        # if num_spks is specified, do onehot coditioners in dec stages
        # subract_mean: from output signal, get rif of mean by windows
        # multilayer_out: add some convs in between gblocks in decoder
        super().__init__(name='Generator1D')
        self.dec_kwidth = dec_kwidth
        self.skip = skip
        self.skip_init = skip_init
        self.skip_dropout = skip_dropout
        self.subtract_mean = subtract_mean
        self.snorm = snorm
        self.z_dim = z_dim
        self.z_all = z_all
        self.pos_code = pos_code
        self.post_skip = post_skip
        self.satt = satt
        self.post_proc = post_proc
        self.onehot = num_spks is not None
        if self.onehot:
            assert num_spks > 0
        self.num_spks = num_spks
        # do not place any z
        self.no_z = no_z
        self.do_cuda = cuda
        self.wd = wd
        self.no_tanh = no_tanh
        self.skip_blacklist = skip_blacklist
        self.gen_enc = nn.ModuleList()
        if aal or aal_out:
            # Make cheby1 filter to include into pytorch conv blocks
            from scipy.signal import cheby1, dlti, dimpulse
            system = dlti(*cheby1(8, 0.05, 0.8 / pooling))
            tout, yout = dimpulse(system)
            filter_h = yout[0]
        if aal:
            self.filter_h = filter_h
        else:
            self.filter_h = None

        if dec_kwidth is None:
            dec_kwidth = kwidth

        if isinstance(activations, str):
            if activations != 'glu':
                activations = getattr(nn, activations)()
        if not isinstance(activations, list):
            activations = [activations] * len(enc_fmaps)
        
        skips = {}
        # Build Encoder
        for layer_idx, (fmaps, act) in enumerate(zip(enc_fmaps, 
                                                     activations)):
            if layer_idx == 0:
                inp = ninputs
            else:
                inp = enc_fmaps[layer_idx - 1]
            self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                       padding=None, lnorm=lnorm, 
                                       dropout=dropout, pooling=pooling,
                                       enc=True, bias=bias, 
                                       aal_h=self.filter_h,
                                       snorm=snorm, convblock=convblock,
                                       satt=self.satt))
        assert dec_fmaps is not None
        dec_inp = enc_fmaps[-1]
        # make summarizer that interpolates each c[n] of the encoder result
        self.enc_int = nn.Sequential(
            nn.ConvTranspose1d(dec_inp, 128, 3, padding=0, output_padding=1,
                               stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(128, 64, 3, padding=0, output_padding=1,
                               stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(64, 32, 3, padding=0, output_padding=1, 
                              stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 10, 3, padding=0, output_padding=1,
                               stride=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(10, 10, 3, padding=0, output_padding=1,
                               stride=4),
            nn.ReLU(inplace=True))
                               

        if pooling != 4:
            raise NotImplementedError

        # Build Decoder
        self.gen_dec = nn.ModuleList()

        if dec_activations is None:
            # assign same activations as in Encoder
            dec_activations = [activations[0]] * len(dec_fmaps)
        
        dec_inp = 11
        dec_fmaps += [1]
        dec_activations += [nn.Tanh()]
        for layer_idx, (fmaps, act) in enumerate(zip(dec_fmaps, 
                                                     dec_activations)):
            if layer_idx >= len(dec_fmaps) - 1:
                lnorm = False
                dropout = 0
            self.gen_dec.append(GBlock(dec_inp,
                                       fmaps, kwidth, act, 
                                       lnorm=lnorm,
                                       dropout=dropout, pooling=1,
                                       padding=kwidth//2,
                                       enc=True,
                                       bias=bias,
                                       convblock=convblock))
            dec_inp = fmaps


        

    def forward(self, x, z=None, ret_hid=False, spkid=None, 
                slice_idx=0, att_weight=0):
        hall = {}
        hi = x
        for l_i, enc_layer in enumerate(self.gen_enc):
            hi, linear_hi = enc_layer(hi, att_weight=att_weight)
            #print('ENC {} hi size: {}'.format(l_i, hi.size()))
                    #print('Adding skip[{}]={}, alpha={}'.format(l_i,
                    #                                            hi.size(),
                    #                                            hi.size(1)))
            if ret_hid:
                hall['enc_{}'.format(l_i)] = hi
        #c = self.enc_summarizer(hi)
        c = self.enc_int(hi)
        z_len = x.size(2)
        if z is None:
            # make z 
            z = torch.randn(x.size(0), 1, z_len)
        else:
            z = z.view(x.size(0), 1, z_len)
        if not hasattr(self, 'z'):
            self.z = z
        # inject c into z
        #c_N = c.size(2)
        #space = z_len // c_N
        #index = torch.LongTensor(list(range(0, c_N * space, space))).view(1, 1,
        #                                                                  -1)
        #index = index.repeat(c.size(0), 1, 1)
        if c.is_cuda:
            z = z.to('cuda')
            #index = index.to('cuda')
        #z.scatter_(2, index, c)
        z = torch.cat((z, c), dim=1)
        if ret_hid:
            hall['enc_zc'] = z
        hi = z
        for l_i, dec_layer in enumerate(self.gen_dec):
            hi, _ = dec_layer(hi, att_weight=att_weight)
            if ret_hid:
                hall['dec_{}'.format(l_i)] = hi
        if ret_hid:
            return hi, hall
        else:
            return hi

if __name__ == '__main__':
    #G = Generator1D(1, [64, 128, 256], 31, 'ReLU',
    #                lnorm=False, dropout=0.5,
    #                pooling=2,
    #                z_dim=256,
    #                z_all=True,
    #                skip_init='randn',
    #                skip_blacklist=[],
    #                bias=True, cuda=False,
    #                rnn_core=False, linterp=False,
    #                dec_kwidth=31)
    G = AttGenerator1D(1, [8, 16, 16, 32, 32, 64, 64, 128, 128, 128, 128], 31, 31,
                       pooling=2,  cuda=False,
                       skip=True)
    print(G)
    x = torch.randn(1, 1, 13000)
    y, hall = G(x, 17, ret_hid=True)
    print(y)
    print(x.size())
    print(y.size())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imshow(hall['att'].data[0, :, :].numpy())
    plt.savefig('att_test.png', dpi=200)
    """
    G = Generator(1, [16, 32, 64, 64, 128, 256, 32, 32, 64, 64, 128, 128, 256, 256], 3, 'ReLU',
                  True, 0.5,
                  z_dim=256,
                  z_all=False,
                  skip_blacklist=[],
                  core2d=True,
                  core2d_felayers=6,
                  bias=True, cuda=True)
    G.parameters()
    G.cuda()
    print(G)
    x = Variable(torch.randn(1, 1, 16384)).cuda()
    y = G(x)
    print(y)
    """
