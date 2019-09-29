import torch
import torch.nn as nn
import random
import torch.nn.utils as nnu
import torch.nn.functional as F
from collections import OrderedDict
from pase.models.modules import SincConv_fast
try:
    from core import Model, LayerNorm
    from modules import *
except ImportError:
    from .core import Model, LayerNorm
    from .modules import *

# BEWARE: PyTorch >= 0.4.1 REQUIRED
from torch.nn.utils.spectral_norm import spectral_norm


def projector_specnorm(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Embedding') != -1:
        spectral_norm(m)

class DBlock(nn.Module):

    def __init__(self, ninp,
                 fmaps,
                 kwidth,
                 stride=1,
                 bias=True,
                 cond_dim=None,
                 norm_type=None):
        super().__init__()
        #fmaps = ninp * stride
        self.stride = stride
        assert kwidth % 2 != 0
        self.conv1 = GConv1DBlock(ninp, fmaps,
                                  kwidth, stride=stride,
                                  bias=bias,
                                  norm_type=norm_type,
                                  pad_mode='constant')
        if cond_dim is not None:
            self.cond_W = nn.Conv1d(cond_dim, fmaps, 1)
        self.conv2 = GConv1DBlock(fmaps, fmaps,
                                  kwidth, stride=1,
                                  bias=bias,
                                  norm_type=norm_type,
                                  pad_mode='constant')
        self.conv3 = nn.Conv1d(fmaps, fmaps, kwidth,
                               padding=kwidth // 2)

        self.W = nn.Conv1d(ninp, fmaps, 1)
        if stride > 1:
            self.downsample = nn.AvgPool1d(stride)

    def forward(self, x, cond=None):
        h1 = self.conv1(x)
        if hasattr(self, 'cond_W') and cond is not None:
            # adapt time resolution if needed
            if cond.shape[2] != h1.shape[2]:
                cond = F.adaptive_avg_pool1d(cond, h1.shape[2])
            ch = self.cond_W(cond)
            h1 = h1 + ch
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        if self.stride > 1:
            x = self.downsample(x)
        res = self.W(x)
        y = h3 + res
        return y


class RWDElement(nn.Module):

    def __init__(self, k,
                 strides,
                 fmaps,
                 kwidth,
                 window=320,
                 cond_level=None,
                 frontend=None,
                 cond_dim=None):
        super().__init__()
        self.k = k
        self.window = window
        self.blocks = nn.ModuleList()
        if frontend is not None:
            assert isinstance(cond_dim, int), type(cond_dim)
            assert isinstance(cond_level, int), type(cond_level)
            self.frontend = frontend
            self.cond_level = cond_level
            self.cond_dim = cond_dim
        ninp = k
        for fmap, stride in zip(fmaps, strides):
            block = DBlock(ninp,
                           fmap,
                           kwidth,
                           stride=stride,
                           cond_dim=cond_dim)
            self.blocks.append(block)
            ninp = fmap

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Conv1d(fmap, 1, 1)
        self.cond_level = cond_level

    def forward(self, x, cond=None):
        # randomly sample a time slice 
        bsz, ch, slen = x.shape
        winlen = self.window * self.k
        ridx = np.random.randint(0, slen - winlen)
        x = x[:, :, ridx:ridx + winlen]
        x = x.view(bsz, ch * self.k, self.window)
        # Process the cond with the frontend if needed
        if hasattr(self, 'frontend'):
            #cond = cond[:, : , ridx:ridx + winlen]
            #print('cond shape: ', cond.shape)
            fe_D = np.cumprod(self.frontend.strides)[-1]
            didx = ridx // fe_D
            cond = self.frontend(cond)
            cond = cond[:, :, didx:didx + (winlen // fe_D)]
        cond_ = None
        for di, dblock in enumerate(self.blocks, 1):
            if di == self.cond_level: 
                cond_ = cond
            else:
                cond_ = None
            x = dblock(x, cond_)
        x = self.pool(x)
        x = self.out(x)
        return x



class RWDiscriminators(Model):

    def __init__(self,
                 k_windows = [1, 2, 4, 8, 15],
                 frontend=None):
        super().__init__()
        self.k_windows = k_windows
        self.frontend = frontend

        if frontend is not None:
            cond_dim = frontend.emb_dim
            cond_level = 4

            self.cond_mods = nn.ModuleList()
            for kwin in self.k_windows:
                rwd = RWDElement(kwin,
                                 [2, 4, 4, 5, 1],
                                 [64, 128, 256, 256, 256],
                                 kwidth=5,
                                 frontend=frontend,
                                 cond_dim=cond_dim,
                                 cond_level=cond_level)
                self.cond_mods.append(rwd)

        self.uncond_mods = nn.ModuleList()
        for kwin in self.k_windows:
            rwd = RWDElement(kwin,
                             [2, 4, 4, 5, 1],
                             [64, 128, 256, 256, 256],
                             kwidth=5)
            self.uncond_mods.append(rwd)

    def forward(self, x, cond=None):
        dout = torch.zeros(x.size(0), 1, 1)
        if cond is not None and self.frontend is not None:
            for cmod in self.cond_mods:
                dcond_ = cmod(x, cond)
                print('Summing cond ', dcond_.item())
                dout = dout + dcond_

        for umod in self.uncond_mods:
            uncond_ = umod(x)
            print('Summing uncond ', uncond_.item())
            dout = uncond_ + dout

        return dout


class DiscriminatorFE(Model):

    def __init__(self, 
                 fmaps=[16, 32, 64, 128, 256, 512],
                 poolings=[4] * 6,
                 kwidths=[16] * 6,
                 frontend=None,
                 cond_dim=100,
                 condkwidth=31,
                 ft_fe=False,
                 bias=False,
                 norm_type='inorm',
                 pool_type='mlp',
                 phase_shift=None,
                 name='DiscriminatorFE'):
        super().__init__(name=name)
        self.phase_shift = phase_shift
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
            enc_block = GConv1DFEBlock(
                ninp, fmap, kw, 
                cond_dim=cond_dim,
                condkwidth=condkwidth,
                stride=pool,
                bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            ninp = fmap

        mlp = [nn.Linear(4 * fmaps[-1], 1024),
               nn.PReLU(1024, init=0)
              ]
        if norm_type == 'bnorm':
            mlp += [nn.BatchNorm1d(1024)]
        mlp += [nn.Linear(1024, 1)]
        self.mlp = nn.Sequential(
            *mlp
        )
        if norm_type == 'snorm':
            torch.nn.utils.spectral_norm(self.mlp[0])
            torch.nn.utils.spectral_norm(self.mlp[-1])

    def forward(self, x, cond):
        # encode noisy
        c = self.frontend(cond)
        hact = {'c':c}
        if not self.ft_fe:
            c = c.detach()
        h = x
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
            h = enc_layer(h, c)
            hact['enc_{}'.format(l_i)] = h
        y = self.mlp(h.view(h.size(0), -1))
        return y, hact

class AcoDiscriminator(Model):
    
    def __init__(self, ninputs, noutputs, 
                 fmaps, kwidth, poolings,
                 norm_type='snorm',
                 aco_level=-1,
                 pool_slen=16,
                 bias=True,
                 phase_shift=None,
                 projectors=[]):
        super().__init__(name='AcoDiscriminator')
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
        self.norm_type = norm_type
        self.enc_blocks = nn.ModuleList()
        self.aco_level = aco_level
        for pi, (fmap, pool) in enumerate(zip(fmaps,
                                              poolings),
                                          start=1):
            enc_block = GConv1DBlock(
                ninp, fmap, kwidth, stride=pool,
                bias=bias,
                norm_type=norm_type
            )
            self.enc_blocks.append(enc_block)
            if pi == aco_level:
                self.build_aco_mlp(fmap, noutputs)
            ninp = fmap
        if not hasattr(self, 'aco_mlp'):
            self.build_aco_mlp(fmap, noutputs)
        # Include projectors into D
        self.projectors = nn.ModuleList()
        for projector in projectors:
            self.projectors.append(projector)
            if norm_type == 'snorm':
                self.projectors[-1].apply(projector_specnorm)
        # Real/fake prediction branch
        # resize tensor to fit into FC directly
        pool_slen *= fmaps[-1]
        self.pool = nn.Sequential(
            nn.Linear(pool_slen, 256),
            nn.PReLU(256)
        )
        self.out_fc = nn.Linear(256, 1)
        if norm_type == 'snorm':
            torch.nn.utils.spectral_norm(self.pool[0])
            torch.nn.utils.spectral_norm(self.out_fc)

    def apply_projectors(self, hid_feat, labs):
        # hid feat is [B, F], it is a must
        assert len(hid_feat.size()) == 2, hid_feat.size()
        outs = []
        for proj, lab in zip(self.projectors, labs):
            outs.append(proj(hid_feat, lab))
        return outs

    def build_aco_mlp(self, ninp, nouts):
        self.aco_mlp = nn.Sequential(nn.Conv1d(ninp, ninp // 2, 1),
                                     nn.PReLU(ninp // 2),
                                     nn.Conv1d(ninp // 2, nouts, 1))
        if self.norm_type == 'snorm':
            torch.nn.utils.spectral_norm(self.aco_mlp[0])
            torch.nn.utils.spectral_norm(self.aco_mlp[2])
    
    def forward(self, x, labs=[], aco_branch=False):
        h = x
        # store intermediate activations
        int_act = {}
        enc_level = 1
        aco_y = None
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
            if enc_level == self.aco_level and aco_branch:
                aco_y = self.aco_mlp(h)
            enc_level += 1
        if aco_branch and aco_y is None:
            # out of the encoder loop, highest hierarchy decimation then
            aco_y = self.aco_mlp(h)
        h = h.view(h.size(0), -1)
        h = self.pool(h)
        cls_y = self.out_fc(h)
        if len(self.projectors) > 0:
            assert len(labs) == len(self.projectors), len(labs)
            y_projs = self.apply_projectors(h, labs)
            for ip, y_proj in enumerate(y_projs):
                int_act['proj_{}'.format(ip)] = y_proj
                # aggregate the projectors outputs
                cls_y += y_proj
        int_act['cls_y'] = cls_y
        int_act['aco_y'] = aco_y
        if aco_y is not None:
            return aco_y, cls_y, int_act
        else:
            return cls_y, int_act


class Discriminator(Model):
    
    def __init__(self, ninputs, fmaps,
                 kwidth, poolings,
                 pool_type='none',
                 pool_slen=None,
                 norm_type='bnorm',
                 bias=True,
                 phase_shift=None, 
                 sinc_conv=False,
                 projectors=[]):
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
            self.sinc_conv = SincConv_fast(1, fmaps[0] // 2,
                                           251, padding='SAME')
            ninp = fmaps[0]
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
        # Include projectors into D
        self.projectors = nn.ModuleList()
        for projector in projectors:
            self.projectors.append(projector)
            if norm_type == 'snorm':
                self.projectors[-1].apply(projector_specnorm)
        self.pool_type = pool_type
        if pool_type == 'none':
            # resize tensor to fit into FC directly
            pool_slen *= fmaps[-1]
            self.pool = nn.Sequential(
                nn.Linear(pool_slen, 256),
                nn.PReLU(256),
            )
            self.out_fc = nn.Linear(256, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.pool[0])
                torch.nn.utils.spectral_norm(self.out_fc)
        elif pool_type == 'conv':
            self.pool = nn.Conv1d(fmaps[-1], 1, 1)
            self.out_fc = nn.Linear(pool_slen, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.pool)
                torch.nn.utils.spectral_norm(self.out_fc)
        elif pool_type == 'gavg':
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.out_fc = nn.Linear(fmaps[-1], 1, 1)
            if norm_type == 'snorm':
                torch.nn.utils.spectral_norm(self.out_fc)
        else:
            raise TypeError('Unrecognized pool type: ', pool_type)

    def apply_projectors(self, hid_feat, labs):
        # hid feat is [B, F], it is a must
        assert len(hid_feat.size()) == 2, hid_feat.size()
        outs = []
        for proj, lab in zip(self.projectors, labs):
            outs.append(proj(hid_feat, lab))
        return outs
    
    def forward(self, x, labs=[]):
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
            h = self.pool(h)
            h = h.view(h.size(0), -1)
            int_act['avg_conv_h'] = h
            y = self.out_fc(h)
        elif self.pool_type == 'none':
            h = h.view(h.size(0), -1)
            h = self.pool(h)
            y = self.out_fc(h)
        elif self.pool_type == 'gavg':
            h = self.pool(h)
            h = h.view(h.size(0), -1)
            y = self.out_fc(h)
        int_act['logit'] = y
        if len(self.projectors) > 0:
            assert len(labs) == len(self.projectors), len(labs)
            y_projs = self.apply_projectors(h, labs)
            for ip, y_proj in enumerate(y_projs):
                int_act['proj_{}'.format(ip)] = y_proj
                # aggregate the projectors outputs
                y += y_proj
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
    from pase.models.frontend import wf_builder
    x = torch.randn(5, 1, 16384)
    fe = wf_builder('../../cfg/PASE.cfg')
    D = DiscriminatorFE(frontend=fe)
    y, h = D(x, x)
    print(D)
    print('y size: ', y.size())
    """
    #projs = [DProjector(256, 80),
    #         DProjector(256, 32)]
    #D = AcoDiscriminator(2, 277, [64, 128, 256, 512, 1024],
    #                    31, [4] * 5, pool_slen=16,
    #                   norm_type='snorm',
    #                   projectors=projs)
    #print(D)
    #x = torch.randn(5, 2, 16384)
    #labs = [torch.ones(5, 1).long(), 
    #        torch.zeros(5, 1).long()]
    #aco_y, y, hact = D(x, labs=labs, aco_branch=True)
    #print(aco_y.size())
    #print(y.size())
    #print(hact.keys())
    from pase.models.frontend import wf_builder
    pase = wf_builder('../../cfg/PASE.cfg')
    #rwd = RWDElement(15, [2, 4, 4, 5, 1],
    #                 [64, 128, 256, 256, 256],
    #                 kwidth=5,
    #                 frontend=pase,
    #                 cond_dim=pase.emb_dim,
    #                 cond_level=4)
    rwds = RWDiscriminators(frontend=pase)
    print(rwds)
    x = torch.randn(1, 1, 16000)
    cond = torch.ones(1, 1, 16000)
    y = rwds(x, cond)
    print(y.shape)
    
