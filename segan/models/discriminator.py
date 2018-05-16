import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _single
try:
    from core import Model, LayerNorm, VirtualBatchNorm1d
except ImportError:
    from .core import Model, LayerNorm, VirtualBatchNorm1d


def l2_norm(x, eps=1e-12):
    return x / (((x**2).sum())**0.5 + eps)

def max_singular_value(W, u=None, Ip=1):
	# https://github.com/godisboy/SN-GAN/blob/master/models/models.py
    """ power iteration for weight parameter """
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        #print(_u.size(), W.size())
        _v = l2_norm(torch.matmul(_u, W.data), eps=1e-12)
        _u = l2_norm(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.matmul(torch.matmul(_v, 
                                      torch.transpose(W.data, 
                                                      0, 
                                                      1)), torch.transpose(_u, 
                                                                           0, 
                                                                           1))
    return sigma, _u

class SNLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = None

    def forward(self, input):
        w_mat = self.weight
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.linear(input, self.weight, self.bias)

class SNConv1d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(SNConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, 
            padding, dilation,
            False, _single(0), groups, bias)
        self.u = None

    def forward(self, input):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.conv1d(input, self.weight, self.bias, self.stride,
						self.padding, self.dilation, self.groups)

class DiscBlock(nn.Module):

    def __init__(self, ninputs, kwidth, nfmaps,
                 activation, bnorm=False, pooling=2, SND=False, 
                 dropout=0):
        super().__init__()
        self.kwidth = kwidth
        seq_dict = OrderedDict()
        if SND:
            self.conv = SNConv1d(ninputs, nfmaps, kwidth,
                                 stride=pooling,
                                 padding=0)
        else:
            self.conv = nn.Conv1d(ninputs, nfmaps, kwidth,
                                  stride=pooling,
                                  padding=0)
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
        x = F.pad(x, ((self.kwidth//2)-1, self.kwidth//2))
        conv_h = self.conv(x)
        if self.bnorm:
            conv_h = self.bn(conv_h)
        conv_h = self.act(conv_h)
        if self.dropout:
            conv_h = self.dout(conv_h)
        return conv_h

class VBNDiscBlock(nn.Module):

    def __init__(self, ninputs, kwidth, nfmaps,
                 activation, pooling=2, SND=False, 
                 dropout=0, cuda=False):
        super().__init__()
        self.kwidth = kwidth
        self.vbnb = nn.ModuleList()
        if SND:
            conv = SNConv1d(ninputs, nfmaps, kwidth,
                            stride=pooling,
                            padding=0)
        else:
            conv = nn.Conv1d(ninputs, nfmaps, kwidth,
                             stride=pooling,
                             padding=0)
        self.vbnb.append(conv)
        vbn = VirtualBatchNorm1d(nfmaps, cuda=cuda)
        self.vbnb.append(vbn)
        if isinstance(activation, str):
            act = getattr(nn, activation)()
        else:
            act = activation
        self.vbnb.append(act)
        if dropout > 0:
            dout = nn.Dropout(dropout)
            self.vbnb.append(dout)

    def forward(self, x, mean=None, mean_sq=None):
        hi = x
        for l in self.vbnb:
            if isinstance(l, nn.Conv1d) or isinstance(l, SNConv1d):
                hi = F.pad(hi, ((self.kwidth//2)-1, self.kwidth//2))
                hi = l(hi)
            elif isinstance(l, VirtualBatchNorm1d):
                hi, mean, mean_sq = l(hi, mean, mean_sq)
            else:
                hi = l(hi)
        return hi, mean, mean_sq

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
        if SND:
            self.bili = SNLinear(8 * d_fmaps[-1], 8 * d_fmaps[-1], bias=True)
        else:
            self.bili = nn.Linear(8 * d_fmaps[-1], 8 * d_fmaps[-1], bias=True)

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
                 dropout=0, Genc=None, pool_size=8, num_spks=None):
        super().__init__(name='Discriminator')
        if Genc is None:
            if not isinstance(activation, list):
                activation = [activation] * len(d_fmaps)
            self.disc = nn.ModuleList()
            for d_i, d_fmap in enumerate(d_fmaps):
                act = activation[d_i]
                if d_i == 0:
                    inp = ninputs
                else:
                    inp = d_fmaps[d_i - 1]
                self.disc.append(DiscBlock(inp, kwidth, d_fmap,
                                           act, bnorm,
                                           pooling, SND,
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
                    nn.Linear(128, outs)
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(pool_size, 256),
                    nn.PReLU(256),
                    nn.Linear(256, 128),
                    nn.PReLU(128),
                    nn.Linear(128, outs)
                )
        elif pool_type == 'rnn':
            if bnorm:
                self.ln = LayerNorm()
            self.rnn = nn.LSTM(d_fmaps[-1], pool_size, batch_first=True,
                               bidirectional=True)
            # bidirectional size
            pool_size *= 2
            self.fc = nn.Linear(pool_size, 1)
        elif pool_type == 'conv':
            self.pool_conv = nn.Conv1d(d_fmaps[-1], 1, 1)
            self.fc = nn.Linear(pool_size, 1)
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
        elif self.pool_type == 'conv':
            h = self.pool_conv(h)
            h = h.view(h.size(0), -1)
            int_act['avg_conv_h'] = h
        elif self.pool_type == 'none':
            h = h.view(h.size(0), -1)
        y = self.fc(h)
        int_act['logit'] = y
        #return F.sigmoid(y), int_act
        return y, int_act

class VBNDiscriminator(Model):
    
    def __init__(self, ninputs, d_fmaps, kwidth, activation,
                 pooling=2, SND=False, pool_size=8, cuda=False):
        super().__init__(name='VBNDiscriminator')
        if not isinstance(activation, list):
            activation = [activation] * len(d_fmaps)
        self.disc = nn.ModuleList()
        for d_i, d_fmap in enumerate(d_fmaps):
            act = activation[d_i]
            if d_i == 0:
                inp = ninputs
            else:
                inp = d_fmaps[d_i - 1]
            self.disc.append(VBNDiscBlock(inp, kwidth, d_fmap,
                                          act, 
                                          pooling, SND,0, 
                                          cuda=cuda))
        self.pool_conv = nn.Conv1d(d_fmaps[-1], 1, 1)
        self.fc = nn.Linear(pool_size, 1)
        self.ref_x = None

    def _forward_conv(self, x, mean=None, mean_sq=None):
        h = x
        means = []
        means_sq = []
        # store intermediate activations
        int_act = {}
        for ii, module in enumerate(self.disc):
            if mean is not None:
                ref_mean = mean[ii]
                ref_mean_sq = mean_sq[ii]
            else:
                ref_mean = ref_mean_sq = None
            h, new_mean, new_mean_sq = module(h, ref_mean,
                                              ref_mean_sq)
            means.append(new_mean)
            means_sq.append(new_mean_sq)
            int_act['h_{}'.format(ii)] = h
        return h, int_act, means, means_sq

    def forward(self, x):
        if self.ref_x is None:
            h, iact, means, means_sq = self._forward_conv(x, None, None)
            # store x
            self.ref_x = x
        else:
            # first pass with ref batch
            ref_hs, ref_iact, \
            ref_means, ref_means_sq = self._forward_conv(self.ref_x, None, None)
            # second pass with real batch
            h, iact, \
            means, means_sq = self._forward_conv(x, ref_means, 
                                                 ref_means_sq)
        h = self.pool_conv(h)
        h = h.view(h.size(0), -1)
        iact['avg_conv_h'] = h
        h = h.view(h.size(0), -1)
        y = self.fc(h)
        iact['logit'] = y
        return y, iact
        #return F.sigmoid(y), iact

if __name__ == '__main__':
    #disc = Discriminator(2, [16, 32, 32, 64, 64, 128, 128, 256, 
    #                         256, 512, 1024], 31, 
    #                     nn.LeakyReLU(0.3))
    disc = BiDiscriminator([16, 32, 32, 64, 64, 128, 128, 256, 
                             256, 512, 1024], 31, 
                         nn.LeakyReLU(0.3))
    print(disc)
    from torch.autograd import Variable
    x = Variable(torch.randn(1, 2, 16384))
    y = disc(x)
    print(y)
