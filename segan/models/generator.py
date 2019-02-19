import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils as nnu
import torch.nn as nn
import random
import numpy as np
try:
    from core import *
    from modules import *
except ImportError:
    from .core import *
    from .modules import *

# BEWARE: PyTorch >= 0.4.1 REQUIRED
from torch.nn.utils.spectral_norm import spectral_norm

class GSkip(nn.Module):

    def __init__(self, skip_type, size, skip_init, skip_dropout=0,
                 merge_mode='sum', kwidth=11, bias=True):
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
            #if cuda:
            #    alpha_ = alpha_.cuda()
            if skip_type == 'alpha':
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
            else:
                # constant, not learnable
                self.skip_k = nn.Parameter(alpha_.view(1, -1, 1))
                self.skip_k.requires_grad = False
        elif skip_type == 'conv':
            if kwidth > 1:
                pad = kwidth // 2
            else:
                pad = 0
            self.skip_k = nn.Conv1d(size, size, kwidth, stride=1,
                                    padding=pad, bias=bias)
        else:
            raise TypeError('Unrecognized GSkip scheme: ', skip_type)
        self.skip_type = skip_type
        if skip_dropout > 0:
            self.skip_dropout = nn.Dropout(skip_dropout)

    def __repr__(self):
        if self.skip_type == 'alpha':
            return self._get_name() + '(Alpha(1))'
        elif self.skip_type == 'constant':
            return self._get_name() + '(Constant(1))'
        else:
            return super().__repr__()

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

class Generator(Model):

    def __init__(self, ninputs, fmaps,
                 kwidth, poolings, 
                 dec_fmaps=None,
                 dec_kwidth=None,
                 dec_poolings=None,
                 z_dim=None,
                 no_z=False,
                 skip=True,
                 bias=False,
                 skip_init='one',
                 skip_dropout=0,
                 skip_type='alpha',
                 norm_type=None,
                 skip_merge='sum',
                 skip_kwidth=11,
                 name='Generator'):
        super().__init__(name=name)
        self.skip = skip
        self.bias = bias
        self.no_z = no_z
        self.z_dim = z_dim
        self.enc_blocks = nn.ModuleList()
        assert isinstance(fmaps, list), type(fmaps)
        assert isinstance(poolings, list), type(poolings)
        if isinstance(kwidth, int): 
            kwidth = [kwidth] * len(fmaps)
        assert isinstance(kwidth, list), type(kwidth)
        skips = {}
        ninp = ninputs
        for pi, (fmap, pool, kw) in enumerate(zip(fmaps, poolings, kwidth),
                                              start=1):
            if skip and pi < len(fmaps):
                # Make a skip connection for all but last hidden layer
                gskip = GSkip(skip_type, fmap,
                              skip_init,
                              skip_dropout,
                              merge_mode=skip_merge,
                              kwidth=skip_kwidth,
                              bias=bias)
                l_i = pi - 1
                skips[l_i] = {'alpha':gskip}
                setattr(self, 'alpha_{}'.format(l_i), skips[l_i]['alpha'])
            enc_block = GConv1DBlock(
                ninp, fmap, kw, stride=pool, bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap

        self.skips = skips
        if not no_z and z_dim is None:
            z_dim = fmaps[-1]
        if not no_z:
            ninp += z_dim
        # Ensure we have fmaps, poolings and kwidth ready to decode
        if dec_fmaps is None:
            dec_fmaps = fmaps[::-1][1:] + [1]
        else:
            assert isinstance(dec_fmaps, list), type(dec_fmaps)
        if dec_poolings is None:
            dec_poolings = poolings[:]
        else:
            assert isinstance(dec_poolings, list), type(dec_poolings)
        self.dec_poolings = dec_poolings
        if dec_kwidth is None:
            dec_kwidth = kwidth[:]
        else:
            if isinstance(dec_kwidth, int): 
                dec_kwidth = [dec_kwidth] * len(dec_fmaps)
        assert isinstance(dec_kwidth, list), type(dec_kwidth)
        # Build the decoder
        self.dec_blocks = nn.ModuleList()
        for pi, (fmap, pool, kw) in enumerate(zip(dec_fmaps, dec_poolings, 
                                                  dec_kwidth),
                                              start=1):
            if skip and pi > 1 and pool > 1:
                if skip_merge == 'concat':
                    ninp *= 2

            if pi >= len(dec_fmaps):
                act = 'Tanh'
            else:
                act = None
            if pool > 1:
                dec_block = GDeconv1DBlock(
                    ninp, fmap, kw, stride=pool,
                    norm_type=norm_type, bias=bias,
                    act=act
                )
            else:
                dec_block = GConv1DBlock(
                    ninp, fmap, kw, stride=1, 
                    bias=bias,
                    norm_type=norm_type
                )
            self.dec_blocks.append(dec_block)
            ninp = fmap

    def forward(self, x, z=None, ret_hid=False):
        hall = {}
        hi = x
        skips = self.skips
        for l_i, enc_layer in enumerate(self.enc_blocks):
            hi, linear_hi = enc_layer(hi, True)
            #print('ENC {} hi size: {}'.format(l_i, hi.size()))
                    #print('Adding skip[{}]={}, alpha={}'.format(l_i,
                    #                                            hi.size(),
                    #                                            hi.size(1)))
            if self.skip and l_i < (len(self.enc_blocks) - 1):
                skips[l_i]['tensor'] = linear_hi
            if ret_hid:
                hall['enc_{}'.format(l_i)] = hi
        if not self.no_z:
            if z is None:
                # make z 
                z = torch.randn(hi.size(0), self.z_dim, *hi.size()[2:])
                if hi.is_cuda:
                    z = z.to('cuda')
            if len(z.size()) != len(hi.size()):
                raise ValueError('len(z.size) {} != len(hi.size) {}'
                                 ''.format(len(z.size()), len(hi.size())))
            if not hasattr(self, 'z'):
                self.z = z
            hi = torch.cat((z, hi), dim=1)
            if ret_hid:
                hall['enc_zc'] = hi
        else:
            z = None
        enc_layer_idx = len(self.enc_blocks) - 1
        for l_i, dec_layer in enumerate(self.dec_blocks):
            if self.skip and enc_layer_idx in self.skips and \
            self.dec_poolings[l_i] > 1:
                skip_conn = skips[enc_layer_idx]
                #hi = self.skip_merge(skip_conn, hi)
                #print('Merging  hi {} with skip {} of hj {}'.format(hi.size(),
                #                                                    l_i,
                #                                                    skip_conn['tensor'].size()))
                hi = skip_conn['alpha'](skip_conn['tensor'], hi)
            #print('DEC in size after skip and z_all: ', hi.size())
            #print('decoding layer {} with input {}'.format(l_i, hi.size()))
            hi = dec_layer(hi)
            #print('decoding layer {} output {}'.format(l_i, hi.size()))
            enc_layer_idx -= 1
            if ret_hid:
                hall['dec_{}'.format(l_i)] = hi
        if ret_hid:
            return hi, hall
        else:
            return hi

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
                 no_z=False,
                 skip_type='alpha', 
                 num_spks=None, multilayer_out=False,
                 skip_merge='sum', snorm=False,
                 convblock=False, post_skip=False,
                 pos_code=False, satt=False,
                 dec_fmaps=None, up_poolings=None,
                 post_proc=False, out_gate=False, 
                 linterp_mode='linear', hidden_comb=False, 
                 big_out_filter=False, z_std=1,
                 freeze_enc=False, skip_kwidth=11,
                 pad_type='constant'):
        # if num_spks is specified, do onehot coditioners in dec stages
        # subract_mean: from output signal, get rif of mean by windows
        # multilayer_out: add some convs in between gblocks in decoder
        super().__init__(name='Generator1D')
        self.dec_kwidth = dec_kwidth
        self.skip_kwidth = skip_kwidth
        self.skip = skip
        self.skip_init = skip_init
        self.skip_dropout = skip_dropout
        self.snorm = snorm
        self.z_dim = z_dim
        self.z_all = z_all
        self.pos_code = pos_code
        self.post_skip = post_skip
        self.big_out_filter = big_out_filter
        self.satt = satt
        self.post_proc = post_proc
        self.pad_type = pad_type
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
        self.freeze_enc = freeze_enc
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
                                  cuda=self.do_cuda,
                                  kwidth=self.skip_kwidth)
                    skips[l_i] = {'alpha':gskip}
                    setattr(self, 'alpha_{}'.format(l_i), skips[l_i]['alpha'])
            self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                       padding=None, lnorm=lnorm, 
                                       dropout=dropout, pooling=pool,
                                       enc=True, bias=bias, 
                                       aal_h=self.filter_h,
                                       snorm=snorm, convblock=convblock,
                                       satt=self.satt,
                                       pad_type=pad_type))
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
                                           comb=hidden_comb,
                                           pad_type=pad_type))
            else:
                self.gen_dec.append(GBlock(dec_inp,
                                           fmaps, dec_kwidth, act, 
                                           lnorm=lnorm,
                                           dropout=dropout, pooling=1,
                                           padding=0,#kwidth//2,
                                           enc=True,
                                           bias=bias,
                                           convblock=convblock,
                                           pad_type=pad_type))
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
        # Cut gradient flow in Encoder?
        if self.freeze_enc:
            hi = hi.detach()
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
        
if __name__ == '__main__':
    """
    G = Generator1D(1, [64, 128, 256, 512, 1024], 
                    31, 
                    'ReLU',
                    lnorm=False, 
                    pooling=4,
                    z_dim=1024,
                    skip_init='randn',
                    skip_type='alpha',
                    skip_blacklist=[],
                    bias=False, cuda=False,
                    rnn_core=False, linterp=False,
                    dec_kwidth=31)
    """
    G = Generator(1, [64, 128, 256, 512, 1024],
                  kwidth=31,
                  poolings=[4, 4, 4, 4, 4], no_z=True)
    print(G)
    print('G num params: ', G.get_n_params())
    x = torch.randn(1, 1, 16384)
    y, hall = G(x, ret_hid=True)
    print(y)
    print(x.size())
    print(y.size())
    #import matplotlib
    #matplotlib.use('Agg')
    #import matplotlib.pyplot as plt
    #plt.imshow(hall['att'].data[0, :, :].numpy())
    #plt.savefig('att_test.png', dpi=200)
