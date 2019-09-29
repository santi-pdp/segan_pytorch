import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
import numpy as np


def build_norm_layer(norm_type, param=None, num_feats=None,
                     num_classes=None):
    if norm_type == 'bnorm':
        if num_classes is not None and num_classes > 1:
            return CategoricalConditionalBatchNorm(num_feats,
                                                   num_classes)
        else: 
            return nn.BatchNorm1d(num_feats)
    elif norm_type == 'inorm':
        return nn.InstanceNorm1d(num_feats, affine=True)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)

def gen_noise(z_size, device='cpu'):
    if device == 'cuda':
        noise = torch.cuda.FloatTensor(*z_size).normal_()
    else:
        noise = torch.FloatTensor(*z_size).normal_()
    return noise

class DProjector(nn.Module):

    def __init__(self, input_dim, num_classes, discrete=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.W = nn.Linear(input_dim, input_dim, bias=False)
        self.discrete = discrete
        if self.discrete:
            self.emb = nn.Embedding(num_classes, input_dim)
        else:
            self.emb = nn.Linear(num_classes, input_dim)

    def forward(self, x, cond_idx):
        # x is [B, F] dim
        if self.discrete:
            # cond_idx contains [B, 1] indexes
            emb = self.emb(cond_idx).squeeze(1)
        else:
            # cond_idx contains [B, num_dims] indexes
            emb = self.emb(cond_idx)
        # emb is [B, F] now, after removing time dim
        proj_emb = self.W(emb)
        cls = torch.bmm(x.unsqueeze(1), proj_emb.unsqueeze(2)).squeeze(2)
        return cls
        

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
                 bias=True, norm_type=None,
                 pad_mode='reflect'):
        super().__init__()
        self.conv = nn.Conv1d(ninp, fmaps, kwidth, stride=stride, bias=bias)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = nn.PReLU(fmaps, init=0)
        self.kwidth = kwidth
        self.stride = stride
        self.pad_mode = pad_mode

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
        x_p = F.pad(x, P, mode=self.pad_mode)
        a = self.conv(x_p)
        a = self.forward_norm(a, self.norm)
        h = self.act(a)
        if ret_linear:
            return h, a
        else:
            return h

class GCondConv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps, kwidth=16,
                 cond_dim=100,
                 condkwidth=31,
                 stride=4, bias=False,
                 norm_type=None,
                 act=None, prelu_init=0):
        super().__init__()
        self.conv = nn.Conv1d(ninp, fmaps,
                              kwidth,
                              stride=stride,
                              padding=0)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.exp_fc = nn.Linear(cond_dim, fmaps * condkwidth)
        if act is not None:
            if isinstance(act, str):
                self.act = getattr(nn, act)()
            else:
                self.act = act
        else:
            self.act = nn.PReLU(fmaps, init=prelu_init)
        self.kwidth = kwidth
        assert condkwidth % 2 != 0, condkwidth
        self.condkwidth = condkwidth
        self.stride = stride
        self.fmaps = fmaps

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x, cond):
        P_ = self.kwidth // 2
        if self.kwidth% 2 == 0:
            x_p = F.pad(x, (P_, P_ - 1), mode='reflect')
        else:
            x_p = F.pad(x, (P_, P_), mode='reflect')
        h = self.conv(x_p)
        avg_cond = torch.mean(cond, dim=2)
        exp_cond = self.exp_fc(avg_cond)
        exp_conds = torch.chunk(exp_cond, exp_cond.size(0), dim=0)
        hs = torch.chunk(h, h.size(0), dim=0)
        out_cond = []
        for hi, ccond_w in zip(hs, exp_conds):
            ccond_w = ccond_w.view(self.fmaps, 1, self.condkwidth)
            hi = F.conv1d(hi, ccond_w, groups=self.fmaps,
                          padding=self.condkwidth // 2)
            out_cond.append(hi)
        h = torch.cat(out_cond, dim=0)
        h = self.act(h)
        h = self.forward_norm(h, self.norm)
        return h

class GResDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=4, 
                 bias=True,
                 norm_type=None,
                 act=None,
                 prelu_init=0.2,
                 drop_last=True,
                 num_classes=None):
        super().__init__()
        pad = max(0, (stride - kwidth)//-2)
        self.lin_W = nn.Conv1d(ninp, fmaps, 1)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps,
                                         kwidth, 
                                         stride=stride,
                                         padding=pad)
        self.hid_act = nn.PReLU(fmaps, init=prelu_init)
        self.norm = build_norm_layer(norm_type, self.deconv,
                                     fmaps, num_classes)
        if act is not None:
            if isinstance(act, str):
                self.act = getattr(nn, act)()
            else:
                self.act = act
        else:
            # Make identity
            self.act = None
        self.kwidth = kwidth
        self.stride = stride
        self.drop_last = drop_last
        self.num_classes = num_classes

    def forward_norm(self, x, norm_layer, lab=None):
        if norm_layer is not None:
            if self.num_classes is not None and self.num_classes > 1:
                return norm_layer(x, lab)
            else:
                return norm_layer(x)
        else:
            return x

    def forward(self, x, lab=None):
        # upsample x linearly first
        up_x = F.interpolate(self.lin_W(x), 
                             scale_factor=self.stride, 
                             mode='linear', 
                             align_corners=True)
        # upsample with learnable params
        h = self.deconv(x)
        if self.kwidth % 2 != 0 and self.drop_last:
            h = h[:, :, :-1]
        h = self.hid_act(h)
        # apply skip connection of linear + non-linear and norm
        h = self.forward_norm(h + up_x, self.norm, lab=lab)
        if self.act is None:
            # linear one
            return h
        # non-linear
        return self.act(h)

class GCondDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps, kwidth,
                 cond_dim=100,
                 condkwidth=31,
                 stride=4, bias=False,
                 norm_type=None,
                 act=None, prelu_init=0):
        super().__init__()
        assert kwidth % 2 == 0, kwidth
        pad = max(0, kwidth - stride) // 2
        self.deconv = nn.ConvTranspose1d(ninp, fmaps,
                                         kwidth,
                                         stride=stride,
                                         padding=pad)
        self.norm = build_norm_layer(norm_type, self.deconv,
                                     fmaps)
        self.exp_fc = nn.Linear(cond_dim, fmaps * condkwidth)
        if act is not None:
            if isinstance(act, str):
                self.act = getattr(nn, act)()
            else:
                self.act = act
        else:
            self.act = nn.PReLU(fmaps, init=prelu_init)
        self.kwidth = kwidth
        assert condkwidth % 2 != 0, condkwidth
        self.condkwidth = condkwidth
        self.stride = stride
        self.fmaps = fmaps

    def forward_norm(self, x, norm_layer):
        if norm_layer is not None:
            return norm_layer(x)
        else:
            return x

    def forward(self, x, cond):
        bsz = x.size(0)
        h = self.deconv(x)
        exp_cond = self.exp_fc(cond)
        exp_conds = exp_cond.view(-1, 1, self.condkwidth)
        hs = h.contiguous().view(1, -1, h.size(-1))
        h = F.conv1d(hs, exp_conds, padding=self.condkwidth // 2,
                     groups=bsz * self.fmaps).view(bsz, self.fmaps, -1)
        h = self.forward_norm(h, self.norm)
        y = self.act(h)
        return y

class GDeconv1DBlock(nn.Module):

    def __init__(self, ninp, fmaps,
                 kwidth, stride=4, 
                 bias=True,
                 norm_type=None,
                 act=None,
                 prelu_init=0.2,
                 drop_last=True,
                 num_classes=None):
        super().__init__()
        pad = max(0, (stride - kwidth)//-2)
        self.deconv = nn.ConvTranspose1d(ninp, fmaps,
                                         kwidth, 
                                         stride=stride,
                                         padding=pad)
        self.norm = build_norm_layer(norm_type, self.deconv,
                                     fmaps, num_classes)
        if act is not None:
            if isinstance(act, str):
                self.act = getattr(nn, act)()
            else:
                self.act = act
        else:
            self.act = nn.PReLU(fmaps, init=prelu_init)
        self.num_classes = num_classes
        self.kwidth = kwidth
        self.stride = stride
        self.drop_last = drop_last

    def forward_norm(self, x, norm_layer, lab=None):
        if norm_layer is not None:
            if self.num_classes is not None and self.num_classes > 1:
                return norm_layer(x, lab)
            else:
                return norm_layer(x)
        else:
            return x

    def forward(self, x, lab=None):
        h = self.deconv(x)
        if self.kwidth % 2 != 0 and self.drop_last and \
           self.stride % 2  == 0:
            h = h[:, :, :-1]
        h = self.forward_norm(h, self.norm, lab=lab)
        h = self.act(h)
        return h

class GResUpsampling(nn.Module):

    def __init__(self, ninp, fmaps,
                 kwidth,
                 cond_dim,
                 stride=1, bias=True,
                 norm_type='bnorm',
                 act=None,
                 prelu_init=0.2):
        super().__init__()
        self.stride = stride
        if stride > 1:
            self.upsample = nn.Upsample(scale_factor=stride)
        self.res = nn.Conv1d(ninp, fmaps, 1)
        if stride > 1:
            # make a deconv layer
            self.deconv = GDeconv1DBlock(ninp, fmaps,
                                         kwidth,
                                         stride=stride,
                                         bias=bias,
                                         norm_type=norm_type,
                                         act=act,
                                         prelu_init=prelu_init)
        else:
            self.deconv = GConv1DBlock(ninp, fmaps, kwidth,
                                       stride=1, bias=bias,
                                       norm_type=norm_type)
        self.cond = HyperCond(fmaps, cond_dim)
        self.conv2 = GConv1DBlock(fmaps, fmaps,
                                  3, stride=1, bias=bias,
                                  norm_type=norm_type)

    def forward(self, x, cond):
        x = x
        h1 = self.deconv(x)
        h2 = self.cond(h1, cond)
        y = self.conv2(h2)
        if hasattr(self, 'upsample'):
            res = self.res(self.upsample(x))
        else:
            res = self.res(x)
        y = y + res
        return y


class HyperCond(nn.Module):
    """ Adapted from https://github.com/joansj/blow/blob/master/src/models/blow.py """

    def __init__(self, fmaps, emb_dim, kwidth=3):
        super().__init__()
        assert kwidth % 2 == 1
        self.fmaps = fmaps
        self.kwidth = kwidth
        self.adapt_w = nn.Linear(emb_dim, fmaps * kwidth)
        self.adapt_b = nn.Linear(emb_dim, fmaps)

    def forward(self, h, emb):
        sbatch, ninp, lchunk=h.size()
        h = h.contiguous()
        # Fast version fully using group convolution
        w = self.adapt_w(emb).view(-1, 1, self.kwidth)
        b = self.adapt_b(emb).view(-1)
        h=torch.nn.functional.conv1d(h.view(1, -1, lchunk),
                                     w, 
                                     bias=b,
                                     padding=self.kwidth//2,
                                     groups=sbatch*ninp).view(sbatch,
                                                              self.fmaps,
                                                              lchunk)
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

def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    """ Based on http://nlp.seas.harvard.edu/2018/04/03/attention.html """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    """ Based on http://nlp.seas.harvard.edu/2018/04/03/attention.html """
    def __init__(self, nheads, hidden_size, dropout=0.):
        super().__init__()
        assert hidden_size % nheads == 0
        self.d_k = hidden_size // nheads
        self.nheads = nheads
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(4)]
        )
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # (1) Linear projections hidden_size -> nheads x d_k
        query, key, value = \
                [l(x).view(nbatches, -1, self.nheads, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key, value))]

        # (2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, dropout=self.dropout)

        # (3) Concat using a view and apply a final linear
        x = x.transpose(1, 2).contiguous() \
                .view(nbatches, -1, self.nheads * self.d_k)
        return self.linears[-1](x), self.attn

class CategoricalConditionalBatchNorm(torch.nn.Module):
    """ Based on
    https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/sn_projection_cgan_64x64_143c.ipynb
    """
    # as in the chainer SN-GAN implementation, we keep per-cat weight and bias
    def __init__(self, num_features, num_cats, eps=2e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_cats = num_cats
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
            self.bias = torch.nn.Parameter(torch.Tensor(num_cats, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()

    def forward(self, input, cats):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        out = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [input.size(0), self.num_features] + (input.dim() - 2) * [1]
            weight = self.weight.index_select(0, cats).view(shape)
            bias = self.bias.index_select(0, cats).view(shape)
            out = out * weight + bias
        return out

    def extra_repr(self):
        return '{num_features}, num_cats={num_cats}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}, ' \
               'track_running_stats=' \
               '{track_running_stats}'.format(**self.__dict__)

class MultiOutput(nn.Module):
    """ Auxiliar multioutput module
        to attach to a Discriminator in the
        end, parsing a config of outputs,
        forwarding data and incorporating
        a loss per output.
    """
    def __init__(self, num_inputs, cfg):
        super().__init__()
        out_blocks = nn.ModuleList()
        losses = []
        for oi, (outk, outv) in enumerate(cfg.items(), start=1):
            print('-' * 30)
            loss = outv['loss']
            nouts = outv['num_outputs']
            print('Building output {} block'.format(outk))
            print('\t>> Loss: {}\n\t>> Units: {:5d}'.format(loss,
                                                            nouts))
            print('-' * 30)
            out_blocks.append(nn.Linear(num_inputs,
                                        nouts))
            losses.append(getattr(nn, loss)())
        self.out_blocks = out_blocks
        self.losses = losses

    def forward(self, x):
        outs = []
        for block in self.out_blocks: 
            outs.append(block(x))
        outs = torch.cat(outs, dim=1)
        return outs

    def loss(self, y_, y):
        tot_loss = 0
        for loss in self.losses:
            closs = loss(y_, y) 
            tot_loss += closs
        return tot_loss


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
    #x = torch.randn(1, 1, 16384)
    #sincnet = SincConv(1024, 251, 16000, padding='SAME')
    #y = sincnet(x)
    #print('y size: ', y.size())
    #x = torch.randn(1, 1, 100)
    #dec = GResDeconv1DBlock(1, 1, 25, stride=5, drop_last=False)
    #y = dec(x)
    #print('y size: ', y.size())
    #x = torch.randn(1, 1024)
    #outs_cfg = {'flag':{'num_outputs':1, 'loss':'BCEWithLogitsLoss'},
    #            'prosody':{'num_outputs':4, 'loss':'MSELoss'},
    #            'lps':{'num_outputs':1025, 'loss':'MSELoss'}
    #           }
    #mo = MultiOutput(1024, outs_cfg)
    #print(mo)
    #y = mo(x)
    #print('y size: ', y.size())
    #print('loss: ', mo.loss(y, torch.zeros(y.size())).item())
    #norm = CategoricalConditionalBatchNorm(10, 100)
    #x = torch.randn(20, 10, 1000)
    #lab = torch.ones(20).long()
    #y = norm(x, lab)
    #print(y.size())
    #print(norm)
    #cond_dec = GCondDeconv1DBlock(1024, 512, 16)
    #cond_dec = GCondConv1DConvBlock(1024, 512, 16)
    x = torch.randn(5, 1024, 100)
    cond = torch.randn(5, 100)
    #print('x size: ', x.size())
    #print('cond size: ', cond.size())
    #y = cond_dec(x, cond)
    #hc = HyperCond(1024, 100)
    #y = hc(x, cond)
    G = nn.ModuleList([
        GResUpsampling(1024, 512, 31, 100, stride=2),
        GResUpsampling(512, 256, 31, 100, stride=4),
        GResUpsampling(256, 128, 31, 100, stride=4),
        GResUpsampling(128, 64, 31, 100, stride=5),
        nn.Conv1d(64, 1, 3, padding=1)
    ])
    print(G)
    for gi, genb in enumerate(G, start=1):
        if gi < len(G):
            x = genb(x, cond)
        else:
            x = genb(x)
        print(x.shape)
    """
    proj = DProjector(2048, 80)
    x = torch.randn(5, 2048)
    cond = torch.ones(5, 1).long()
    y = proj(x, cond)
    """

