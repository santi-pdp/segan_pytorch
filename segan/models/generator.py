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
                 pooling=2, enc=True):
        super().__init__()
        if padding is None:
            padding = (kwidth // 2)
        if enc:
            self.conv = nn.Conv1d(ninputs, fmaps, kwidth,
                                  stride=pooling,
                                  padding=padding)
        else:
            # decoder like with transposed conv
            self.conv = nn.ConvTranspose1d(ninputs, fmaps, kwidth,
                                           stride=pooling,
                                           padding=padding)
        if bnorm:
            self.bn = nn.BatchNorm1d(fmaps)
        self.act = activation
        if dropout > 0:
            self.dout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.conv(x)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        h = self.act(h)
        if hasattr(self, 'dout'):
            h = self.dout(h)
        return h

class Generator(Model):

    def __init__(self, ninputs, enc_fmaps, kwidth, 
                 activations, bnorm=False, dropout=0.,
                 pooling=2, z_dim=1024, z_all=False,
                 skip=True, dec_activations=None, cuda=False):
        super().__init__(name='Generator')
        self.skip = skip
        self.z_dim = z_dim
        self.z_all = z_all
        self.do_cuda = cuda
        self.gen_enc = nn.ModuleList()
        if isinstance(activations, str):
            activations = getattr(nn, activations)()
        if not isinstance(activations, list):
            activations = [activations] * len(enc_fmaps)

        for layer_idx, (fmaps, act) in enumerate(zip(enc_fmaps, 
                                                     activations)):
            if layer_idx == 0:
                inp = ninputs
            else:
                inp = enc_fmaps[layer_idx - 1]
            self.gen_enc.append(GBlock(inp, fmaps, kwidth, act,
                                       None, bnorm, dropout, pooling))

        dec_inp = enc_fmaps[-1]
        dec_fmaps = enc_fmaps[::-1][1:] + [1]
        #print('dec_fmaps: ', dec_fmaps)
        self.gen_dec = nn.ModuleList()
        if dec_activations is None:
            dec_activations = activations
        
        dec_inp += z_dim

        for layer_idx, (fmaps, act) in enumerate(zip(dec_fmaps, 
                                                     dec_activations)):
            if skip and layer_idx > 0:
                dec_inp *= 2

            if z_all and layer_idx > 0:
                dec_inp += z_dim

            if layer_idx >= len(dec_fmaps) - 1:
                act = nn.Tanh()
                bnorm = False
                dropout = 0

            self.gen_dec.append(GBlock(dec_inp,
                                       fmaps, kwidth + 1, act, 
                                       kwidth//2, bnorm,
                                       dropout, pooling, False))
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
        #print('=' * 50)
        skips = skips[::-1]
        if z is None:
            # make 1 z and repeat in time
            z = Variable(torch.randn(x.size(0), self.z_dim)).unsqueeze(2)
            z = z.repeat(1,1,hi.size(2))
        if len(z.size()) == 2:
            z = z.unsqueeze(2).repeat(1, 1, hi.size(2))
        if self.do_cuda:
            z = z.cuda()
        hi = torch.cat((hi, z), dim=1)
        #print('Enc out size: ', hi.size())
        z_up = z
        for l_i, dec_layer in enumerate(self.gen_dec):
            #print('DEC in size: ', hi.size())
            if self.skip and l_i > 0:
                skip_conn = skips[l_i - 1]
                #print('concating skip {} to hi {}'.format(skip_conn.size(),
                                                          #hi.size()))
                hi = torch.cat((hi, skip_conn), dim=1)
            if l_i > 0 and self.z_all:
                # concat z in every layer
                #print('z.size: ', z.size())
                z_up = torch.cat((z_up, z_up), dim=2)
                hi = torch.cat((hi, z_up), dim=1)
            hi = dec_layer(hi)
            #print('-' * 20)
            #print('hi size: ', hi.size())
        return hi


if __name__ == '__main__':
    G = Generator(1, [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 10], 31, 'ReLU',
                  True, 0.5,
                  z_all=True)
    print(G)
    x = Variable(torch.randn(1, 1, 16384))
    y = G(x)
    print(y)

