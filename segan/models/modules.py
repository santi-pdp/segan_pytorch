import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
import numpy as np


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
                 kwidth, dilation=1, bias=True, norm_type=None,
                 hid_act=nn.ReLU(inplace=True),
                 out_act=None,
                 skip_init=0):
        super().__init__()
        # first conv level to expand/compress features
        self.entry_conv = nn.Conv1d(num_inputs, hidden_size, 1, bias=bias)
        self.entry_norm = build_norm_layer(norm_type, self.entry_conv, hidden_size)
        self.entry_act = hid_act
        # second conv level to exploit temporal structure
        self.mid_conv = nn.Conv1d(hidden_size, hidden_size, kwidth,
                                  dilation=dilation, bias=bias)
        self.mid_norm = build_norm_layer(norm_type, self.mid_conv, hidden_size)
        self.mid_act = hid_act
        # third conv level to expand/compress features back
        self.exit_conv = nn.Conv1d(hidden_size, num_inputs, 1, bias=bias)
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
                 kwidth, stride=1, 
                 bias=True, norm_type=None):
        super().__init__()
        self.conv = nn.Conv1d(ninp, fmaps, kwidth, stride=stride, bias=bias)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x, ret_linear=False):
        if self.stride > 1:
            P = (self.kwidth // 2 - 1,
                 self.kwidth // 2)
        else:
            P = (self.kwidth // 2,
                 self.kwidth // 2)
        x_p = F.pad(x, P, mode='reflect')
        a = self.conv(x_p)
        a = self.forward_norm(a, self.norm)
        h = self.act(a)
        if ret_linear:
            return h, a
        else:
            return h

class GDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=4, 
                 bias=True,
                 norm_type=None,
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
                 bias=True,
                 norm_type=None,
                 act=None):
        super().__init__()
        self.dil_conv = nn.Conv1d(ninp, fmaps,
                                  kwidth, dilation=dilation,
                                  bias=bias)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv,
                                         fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        # skip 1x1 convolution
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1, bias=bias)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, 
                                                   self.conv_1x1_skip,
                                                   ninp)
        # residual 1x1 convolution
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1, bias=bias)
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
        h = self.forward_norm(h, self.conv_1x1_skip_norm)
        # return with skip connection
        y = x + h
        # also return res connection (going to further net point directly)
        sh = self.conv_1x1_res(a)
        sh = self.forward_norm(sh, self.conv_1x1_res_norm)
        return y, sh

# SincNet conv layer
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band,t_right, cuda=False):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    ones = torch.ones(1)
    if cuda:
        ones = ones.to('cuda')
    y=torch.cat([y_left, ones, y_right])

    return y
    
    
# Modified from https://github.com/mravanelli/SincNet
class SincConv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs,
                 padding='VALID'):
        super(SincConv, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) \
                                         / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, 
                                 N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1)) # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100
                
        self.freq_scale=fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1/self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2-b1)/self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.padding = padding
        
    def forward(self, x):
        cuda = x.is_cuda
        filters=torch.zeros((self.N_filt, self.Filt_dim))
        N=self.Filt_dim
        t_right=torch.linspace(1, (N - 1) / 2, 
                               steps=int((N - 1) / 2)) / self.fs
        if cuda:
            filters = filters.to('cuda')
            t_right = t_right.to('cuda')
        
        min_freq=50.0;
        min_band=50.0;
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + \
                                         min_band / self.freq_scale)
        n = torch.linspace(0, N, steps = N)
        # Filter window (hamming)
        window=(0.54 - 0.46 * torch.cos(2 * math.pi * n / N)).float()
        if cuda:
            window = window.to('cuda')
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float()* \
                    sinc(filt_beg_freq[i].float() * self.freq_scale, 
                         t_right, cuda)
            low_pass2 = 2 * filt_end_freq[i].float()* \
                    sinc(filt_end_freq[i].float() * self.freq_scale, 
                         t_right, cuda)
            band_pass=(low_pass2 - low_pass1)
            band_pass=band_pass/torch.max(band_pass)
            if cuda:
                band_pass = band_pass.to('cuda')

            filters[i,:]=band_pass * window
        if self.padding == 'SAME':
            x_p = F.pad(x, (self.Filt_dim // 2,
                            self.Filt_dim // 2), mode='reflect')
        else:
            x_p = x
        out = F.conv1d(x_p, filters.view(self.N_filt, 1, self.Filt_dim))
        return out

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


if __name__ == '__main__':
    """
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
    """
    x = torch.randn(1, 1, 16384)
    sincnet = SincConv(1024, 251, 16000, padding='SAME')
    y = sincnet(x)
    print('y size: ', y.size())



