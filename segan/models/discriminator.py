import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _single
try:
    from core import Model
except ImportError:
    from .core import Model


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
        seq_dict = OrderedDict()
        if SND:
            conv = SNConv1d(ninputs, nfmaps, kwidth,
                            stride=pooling,
                            padding=(kwidth // 2))
        else:
            conv = nn.Conv1d(ninputs, nfmaps, kwidth,
                             stride=pooling,
                             padding=(kwidth // 2))
        seq_dict['conv'] = conv
        if bnorm:
            bn = nn.BatchNorm1d(nfmaps)
            seq_dict['bn'] = bn
        if isinstance(activation, str):
            act = getattr(nn, activation)()
        else:
            act = activation
        seq_dict['act'] = act
        if dropout > 0:
            seq_dict['dout'] = nn.Dropout(dropout)
        self.block = nn.Sequential(seq_dict)

    def forward(self, x):
        return self.block(x)


class Discriminator(Model):
    
    def __init__(self, ninputs, d_fmaps, kwidth, activation,
                 bnorm=False, pooling=2, SND=False, rnn_pool=False,
                 dropout=0, rnn_size=8):
        super().__init__(name='Discriminator')
        self.disc = nn.ModuleList()
        for d_i, d_fmap in enumerate(d_fmaps):
            if d_i == 0:
                inp = ninputs
            else:
                inp = d_fmaps[d_i - 1]
            self.disc.append(DiscBlock(inp, kwidth, d_fmap,
                                       activation, bnorm,
                                       pooling, SND,
                                       dropout))
        if rnn_pool:
            self.rnn = nn.LSTM(d_fmaps[-1], rnn_size, batch_first=True)
        else:
            self.disc.append(nn.Conv1d(d_fmaps[-1], 1, 1))
        self.fc = nn.Linear(rnn_size, 1)
    
    def forward(self, x):
        h = x
        for layer in self.disc:
            h = layer(h)
        if hasattr(self, 'rnn'):
            ht, state = self.rnn(h.transpose(1,2))
            h = state[0].squeeze(0)
        else:
            h = h.view(h.size(0), -1)
        y = self.fc(h)
        return y


if __name__ == '__main__':
    disc = Discriminator(2, [16, 32, 32, 64, 64, 128, 128, 256, 
                             256, 512, 1024], 31, 
                         nn.LeakyReLU(0.3))
    print(disc)
    from torch.autograd import Variable
    x = Variable(torch.randn(1, 2, 16384))
    y = disc(x)
    print(y)
