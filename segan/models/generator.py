import torch
from torch.autograd import Variable
import torch.nn as nn
try:
    from core import Model
except ImportError:
    from .core import Model

class GBlock(nn.Module):

    def __init__(self, ninputs, fmaps, kwidth,
                 activation, padding=None,
                 bnorm=False, dropout=0.,
                 pooling=2, enc=True, bias=False,
                 aal_h=None):
        super().__init__()
        if padding is None:
            padding = (kwidth // 2)
        if enc:
            if aal_h is not None:
                self.aal_conv = nn.Conv1d(ninputs, ninputs, 
                                          aal_h.shape[0],
                                          stride=1,
                                          padding=aal_h.shape[0] // 2 - 1,
                                          bias=False)
                # apply AAL weights, reshaping impulse response to match
                # in channels and out channels
                aal_t = torch.FloatTensor(aal_h).view(1, 1, -1)
                aal_t = aal_t.repeat(ninputs, ninputs, 1)
                self.aal_conv.weight.data = aal_t
            self.conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                  stride=pooling,
                                  padding=padding,
                                  bias=bias)
        else:
            # decoder like with transposed conv
            self.conv = nn.ConvTranspose1d(ninputs, fmaps, kwidth,
                                           stride=pooling,
                                           padding=padding)
        if bnorm:
            self.bn = nn.BatchNorm1d(fmaps)
        if activation is not None:
            self.act = activation
        if dropout > 0:
            self.dout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.size()) == 4:
            # inverse case from 1D -> 2D, go 2D -> 1D
            # re-format input from [B, K, C, L] to [B, K * C, L]
            # where C: frequency, L: time
            x = x.squeeze(1)
        if hasattr(self, 'aal_conv'):
            x = self.aal_conv(x)
        h = self.conv(x)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'act'):
            h = self.act(h)
        if hasattr(self, 'dout'):
            h = self.dout(h)
        return h


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

class Generator(Model):

    def __init__(self, ninputs, enc_fmaps, kwidth, 
                 activations, bnorm=False, dropout=0.,
                 pooling=2, z_dim=1024, z_all=False,
                 skip=True, skip_blacklist=[],
                 dec_activations=None, cuda=False,
                 bias=False, aal=False, wd=0.,
                 core2d=False, skip_mode='concat'):
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
            if layer_idx == 0 or not core2d:
                self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                           padding=None, bnorm=bnorm, 
                                           dropout=dropout, pooling=pooling,
                                           enc=True, bias=bias, 
                                           aal_h=self.filter_h))
            else:
                if layer_idx == 1:
                    # fmaps is 1 after first conv1d
                    inp = 1
                self.gen_enc.append(G2Block(inp, fmaps, kwidth, act,
                                            padding=None, bnorm=bnorm, 
                                            dropout=dropout, pooling=pooling,
                                            enc=True, bias=bias))
        dec_inp = enc_fmaps[-1]
        if self.core2d:
            #dec_fmaps = enc_fmaps[::-1][1:-2]+ [1, 1]
            dec_fmaps = enc_fmaps[::-1][:-2] + [1, 1]
        else:
            dec_fmaps = enc_fmaps[::-1][1:]+ [1] 
        print(dec_fmaps)
        print(enc_fmaps)
        #print('dec_fmaps: ', dec_fmaps)
        self.gen_dec = nn.ModuleList()
        if dec_activations is None:
            dec_activations = activations
        
        dec_inp += z_dim

        for layer_idx, (fmaps, act) in enumerate(zip(dec_fmaps, 
                                                     dec_activations)):
            if skip and layer_idx > 0 and layer_idx not in skip_blacklist:
                print('Adding skip conn input of idx: {} and size:'
                      ' {}'.format(layer_idx, dec_inp))
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
                                            fmaps, kwidth + 1, act, 
                                            padding=kwidth//2, 
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
        #print('z size: ', z.size())
        hi = torch.cat((hi, z), dim=1)
        #print('Input to dec after concating z and enc out: ', hi.size())
        #print('Enc out size: ', hi.size())
        z_up = z
        for l_i, dec_layer in enumerate(self.gen_dec):
            print('dec layer: {} with input: {}'.format(l_i, hi.size()))
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


if __name__ == '__main__':
    G = Generator(1, [256, 32, 32, 64, 64, 128, 128, 256, 256], 3, 'ReLU',
                  True, 0.5,
                  z_dim=256,
                  z_all=False,
                  skip_blacklist=[],
                  core2d=False,
                  bias=True, cuda=True)
    G.parameters()
    G.cuda()
    print(G)
    x = Variable(torch.randn(1, 1, 16384)).cuda()
    y = G(x)
    print(y)

