import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm


def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)

class ResBlock1D(nn.Module):

    def __init__(self, num_inputs, hidden_size,
                 kwidth, dilation=1, norm_type=None,
                 hid_act=nn.ReLU(inplace=True),
                 out_act=None,
                 skip_init=0):
        super().__init__()
        # first conv level to expand/compress features
        self.entry_conv = nn.Conv1d(num_inputs, hidden_size, 1)
        self.entry_norm = build_norm_layer(norm_type, self.entry_conv, hidden_size)
        self.entry_act = hid_act
        # second conv level to exploit temporal structure
        self.mid_conv = nn.Conv1d(hidden_size, hidden_size, kwidth,
                                  dilation=dilation)
        self.mid_norm = build_norm_layer(norm_type, self.mid_conv, hidden_size)
        self.mid_act = hid_act
        # third conv level to expand/compress features back
        self.exit_conv = nn.Conv1d(hidden_size, num_inputs, 1)
        self.exit_norm = build_norm_layer(norm_type, self.exit_conv, num_inputs)
        if out_act is None:
            out_act = hid_act
        self.exit_act = out_act
        self.kwidth = kwidth
        self.dilation = dilation
        self.skip_alpha = nn.Parameter(torch.FloatTensor([skip_init]))

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        # entry level
        h = self.entry_conv(x)
        h = self.forward_norm(h, self.entry_norm)
        h = self.entry_act(h)
        # mid level
        # first padding
        kw_2 = self.kwidth // 2
        P = kw_2 + kw_2 * (self.dilation - 1)
        h_p = F.pad(h, (P, P), mode='reflect')
        h = self.mid_conv(h_p)
        h = self.forward_norm(h, self.mid_norm)
        h = self.mid_act(h)
        # exit level
        h = self.exit_conv(h)
        h = self.forward_norm(h, self.exit_norm)
        # skip connection + exit_act
        y = self.exit_act(self.skip_alpha * x + h)
        return y

class GConv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=1, norm_type=None):
        super().__init__()
        self.conv = nn.Conv1d(ninp, fmaps, kwidth, stride=stride)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        if self.stride > 1:
            P = (self.kwidth // 2 - 1,
                 self.kwidth // 2)
        else:
            P = (self.kwidth // 2,
                 self.kwidth // 2)
        x_p = F.pad(x, P, mode='reflect')
        h = self.conv(x_p)
        h = self.forward_norm(h, self.norm)
        h = self.act(h)
        return h

class GDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=4, norm_type=None,
                 act=None):
        super().__init__()
        pad = max(0, (stride - kwidth)//-2)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps,
                                         kwidth, 
                                         stride=stride,
                                         padding=pad)
        self.norm = build_norm_layer(norm_type, self.deconv,
                                     fmaps)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        h = self.deconv(x)
        if self.kwidth % 2 != 0:
            h = h[:, :, :-1]
        h = self.forward_norm(h, self.norm)
        h = self.act(h)
        return h

class ResARModule(nn.Module):

    def __init__(self, ninp, fmaps,
                 res_fmaps,
                 kwidth, dilation,
                 norm_type=None,
                 act=None):
        super().__init__()
        self.dil_conv = nn.Conv1d(ninp, fmaps,
                                  kwidth, dilation=dilation)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv,
                                         fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        # skip 1x1 convolution
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, 
                                                   self.conv_1x1_skip,
                                                   ninp)
        # residual 1x1 convolution
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1)
        self.conv_1x1_res_norm = build_norm_layer(norm_type, 
                                                  self.conv_1x1_res,
                                                  res_fmaps)

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x):
        kw__1 = self.kwidth - 1
        P = kw__1 + kw__1 * (self.dilation - 1)
        # causal padding
        x_p = F.pad(x, (P, 0))
        # dilated conv
        h = self.dil_conv(x_p)
        # normalization if applies
        h = self.forward_norm(h, self.dil_norm)
        # activation
        h = self.act(h)
        a = h
        # conv 1x1 to make residual connection
        h = self.conv_1x1_skip(h)
        # normalization if applies
        h = self.forward_norm(h, self.conv_1x1_norm)
        # return with skip connection
        y = x + h
        # also return res connection (going to further net point directly)
        sh = self.conv_1x1_res(a)
        sh = self.forward_norm(sh, self.conv_1x1_res_norm)
        return y, sh


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    # 800 samples @ 16kHz is 50ms
    T = 800
    # n = 20 z time-samples per frame
    n = 20
    zgen = ZGen(n, T // n, 
                z_amp=0.5)
    all_z = None
    for t in range(0, 200, 5):
        time_idx = torch.LongTensor([t])
        z_ten = zgen(time_idx)
        print(z_ten.size())
        z_ten = z_ten.squeeze()
        if all_z is None:
            all_z = z_ten
        else:
            all_z = np.concatenate((all_z, z_ten), axis=1)
    N = 20
    for k in range(N):
        plt.subplot(N, 1, k + 1)
        plt.plot(all_z[k, :], label=k)
        plt.ylabel(k)
    plt.show()

    # ResBlock
    resblock = ResBlock1D(40, 100, 5, dilation=8)
    print(resblock)
    z = z_ten.unsqueeze(0)
    print('Z size: ', z.size())
    y = resblock(z)
    print('Y size: ', y.size())

    x = torch.randn(1, 1, 16) 
    deconv = GDeconv1DBlock(1, 1, 31)
    y = deconv(x)
    print('x: {} -> y: {} deconv'.format(x.size(),
                                         y.size()))
    conv = GConv1DBlock(1, 1, 31, stride=4)
    z = conv(y)
    print('y: {} -> z: {} conv'.format(y.size(),
                                       z.size()))



