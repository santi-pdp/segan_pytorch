import torch
import torch.nn as nn
from random import shuffle
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim import lr_scheduler
from ..datasets import *
from ..utils import *
from .ops import *
from scipy.io import wavfile
import multiprocessing as mp
import numpy as np
import timeit
import random
from random import shuffle
from tensorboardX import SummaryWriter
from .generator import *
from .discriminator import *
from .core import *
import json
import os
from torch import autograd
from scipy import signal


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        print('Initializing weights of convresblock to 0.0, 0.02')
        for k, p in m.named_parameters():
            if 'weight' in k and 'conv' in k:
                p.data.normal_(0.0, 0.02)
    elif classname.find('Conv1d') != -1:
        print('Initialzing weight to 0.0, 0.02 for module: ', m)
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            print('bias to 0 for module: ', m)
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        print('Initializing FC weight to xavier uniform')
        nn.init.xavier_uniform_(m.weight.data)

def wsegan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1DResBlock') != -1:
        print('Initializing weights of convresblock to 0.0, 0.02')
        for k, p in m.named_parameters():
            if 'weight' in k and 'conv' in k:
                nn.init.xavier_uniform_(p.data)
    elif classname.find('Conv1d') != -1:
        print('Initialzing weight to XU for module: ', m)
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('ConvTranspose1d') != -1:
        print('Initialzing weight to XU for module: ', m)
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find('Linear') != -1:
        print('Initializing FC weight to XU')
        nn.init.xavier_uniform_(m.weight.data)

def z_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        # let it active
        m.train()
    else:
        m.eval()


class SEGAN(Model):

    def __init__(self, opts, name='SEGAN',
                 generator=None,
                 discriminator=None):
        super(SEGAN, self).__init__(name)
        self.opts = opts
        self.preemph = opts.preemph
        self.save_path = opts.save_path
        self.do_cuda = opts.cuda
        self.z_dim = opts.z_dim
        self.g_enc_fmaps = opts.g_enc_fmaps
        self.pooling = opts.pooling_size
        if hasattr(opts, 'g_snorm'):
            self.g_snorm = opts.g_snorm
        else:
            self.g_snorm = False
        if hasattr(opts, 'SND'):
            self.SND = opts.SND
        else:
            self.SND = False
        if hasattr(opts, 'd_bnorm'):
            self.d_bnorm = opts.d_bnorm
        else:
            self.d_bnorm = False
        if hasattr(opts, 'g_lnorm'):
            self.g_lnorm = opts.g_lnorm
        else:
            self.g_lnorm = False
        if hasattr(opts, 'linterp'):
            self.linterp = opts.linterp
        else:
            self.linterp = False
        if hasattr(opts, 'convblock'):
            self.convblock = opts.convblock
        else:
            self.convblock = False
        if hasattr(opts, 'dkwidth') and opts.dkwidth is not None:
            # disc kwidth
            self.dkwidth = opts.dkwidth
        else:
            self.dkwidth = opts.kwidth
        if hasattr(opts, 'deckwidth') and opts.deckwidth is not None:
            self.deckwidth = opts.deckwidth
        else:
            self.deckwidth = opts.kwidth
        if hasattr(opts, 'dpooling_size') and opts.dpooling_size is not None:
            self.dpooling_size = opts.dpooling_size
        else:
            self.dpooling_size = opts.pooling_size
        if hasattr(opts, 'd_pool_type'):
            self.d_pool_type = opts.d_pool_type
        else:
            self.d_pool_type = 'conv'
        if hasattr(opts, 'skip_kwidth'):
            self.skip_kwidth = opts.skip_kwidth
        else:
            self.skip_kwidth = 11
        if hasattr(opts, 'post_skip'):
            self.post_skip = opts.post_skip
        else:
            self.post_skip = False
        if hasattr(opts, 'canvas_l2'):
            self.canvas_l2 = opts.canvas_l2
        else:
            self.canvas_l2 = 0.
        if hasattr(opts, 'freeze_genc'):
            self.freeze_genc = opts.freeze_genc
        else:
            self.freeze_genc = False

        self.z_dropout = False
        self.no_z = False
        self.g_dropout = 0.
        if hasattr(opts, 'z_dropout'):
            # dropout is the sampling method
            self.z_dropout = opts.z_dropout
            if self.z_dropout:
                self.no_z = True
                self.g_dropout = 0.2
        if hasattr(opts, 'no_z'):
            self.no_z = opts.no_z
        if hasattr(opts, 'z_std'):
            self.z_std = opts.z_std
        else:
            self.z_std = 1
        if hasattr(opts, 'no_skip'):
            self.no_skip = opts.no_skip
        else:
            self.no_skip = False
        if hasattr(opts, 'pos_code'):
            self.pos_code = opts.pos_code
        else:
            self.pos_code = False
        if hasattr(opts, 'satt'):
            self.satt = opts.satt
        else:
            self.satt = False
        if hasattr(opts, 'mlpconv'):
            self.mlpconv = opts.mlpconv
        else:
            self.mlpconv = False
        if hasattr(opts, 'phase_shift'):
            self.phase_shift = opts.phase_shift
        else:
            self.phase_shift = None
        if hasattr(opts, 'g_dec_fmaps'):
            self.g_dec_fmaps = opts.g_dec_fmaps
        else:
            self.g_dec_fmaps = None
        if hasattr(opts, 'up_poolings'):
            self.up_poolings = opts.up_poolings
        else:
            self.up_poolings = None
        if hasattr(opts, 'comb_net'):
            self.comb_net = opts.comb_net
        else:
            self.comb_net = False
        if hasattr(opts, 'out_gate'):
            self.out_gate = opts.out_gate
        else:
            self.out_gate = False
        if hasattr(opts, 'linterp_mode'):
            self.linterp_mode = opts.linterp_mode
        else:
            self.linterp_mode = 'linear'
        if hasattr(opts, 'hidden_comb'):
            self.hidden_comb = opts.hidden_comb
        else:
            self.hidden_comb = False
        if hasattr(opts, 'pad_type'):
            self.pad_type = opts.pad_type
        else:
            self.pad_type = 'constant'
        if hasattr(opts, 'big_out_filter'):
            self.big_out_filter = opts.big_out_filter
        else:
            self.big_out_filter = False
        if hasattr(opts, 'sinc_conv'):
            self.sinc_conv = opts.sinc_conv
        else:
            self.sinc_conv = False
        if hasattr(opts, 'bias'):
            self.bias = True
        else:
            self.bias = False

        if opts.g_act == 'prelu':
            self.g_enc_act = [nn.PReLU(fmaps) for fmaps in self.g_enc_fmaps]
            self.g_dec_act = [nn.PReLU(fmaps) for fmaps in \
                                 self.g_enc_fmaps[::-1][1:] + [1]]
        elif opts.g_act == 'tanh':
            self.g_enc_act = 'Tanh'
            self.g_dec_act = None
        elif opts.g_act == 'relu':
            self.g_enc_act = 'ReLU'
            self.g_dec_act = None
        else:
            raise TypeError('Unrecognized G activation: ', opts.g_act)
        if generator is None:
            # Build G and D
            self.G = Generator1D(1, 
                                 self.g_enc_fmaps, 
                                 opts.kwidth,
                                 self.g_enc_act,
                                 pooling=opts.pooling_size,
                                 z_dim=self.g_enc_fmaps[-1],
                                 cuda=opts.cuda,
                                 mlpconv=self.mlpconv,
                                 skip=(not self.no_skip),
                                 lnorm=self.g_lnorm,
                                 dropout=self.g_dropout,
                                 no_z=self.no_z,
                                 pos_code=self.pos_code,
                                 dec_activations=self.g_dec_act,
                                 bias=self.bias,
                                 skip_init=opts.skip_init,
                                 dec_kwidth=self.deckwidth,
                                 skip_type=opts.skip_type,
                                 skip_merge=opts.skip_merge,
                                 snorm=self.g_snorm, 
                                 linterp=self.linterp,
                                 convblock=self.convblock,
                                 post_skip=self.post_skip,
                                 satt=self.satt, 
                                 dec_fmaps=self.g_dec_fmaps,
                                 up_poolings=self.up_poolings,
                                 post_proc=self.comb_net,
                                 out_gate=self.out_gate,
                                 linterp_mode=self.linterp_mode,
                                 hidden_comb=self.hidden_comb,
                                 z_std=self.z_std,
                                 freeze_enc=self.freeze_genc,
                                 skip_kwidth=self.skip_kwidth,
                                 pad_type=self.pad_type)

        else:
            self.G = generator
        self.G.apply(weights_init)
        print('Generator: ', self.G)

        self.d_enc_fmaps = opts.d_enc_fmaps
        if discriminator is None:
            self.D = Discriminator(2, self.d_enc_fmaps, self.dkwidth,
                                   nn.LeakyReLU(0.3), 
                                   bnorm=self.d_bnorm,
                                   pooling=self.dpooling_size,
                                   pool_type=self.d_pool_type,
                                   pool_size=opts.D_pool_size, 
                                   SND=self.SND,
                                   phase_shift=self.phase_shift,
                                   sinc_conv=self.sinc_conv)
        else:
            self.D = discriminator
        self.D.apply(weights_init)
        print('Discriminator: ', self.D)
        if self.do_cuda:
            self.D.cuda()
            self.G.cuda()

    def load_raw_weights(self, raw_weights_dir):
        # TODO: get rid of this, it was just used for testing stuff
        # test to load raw weights from TF model and check possibles
        # differences in performance because of architecture
        # weights have fname 'g_ae_dec_5_W'
        # biases have fname 'g_ae_dec_5_b'
        # alphas prelu fname g_ae_dec_prelu_0_alpha
        for l_i, g_enc_l in enumerate(self.G.gen_enc):
            # read weights and biases
            w_npy = np.load(os.path.join(os.path.join(raw_weights_dir,
                                                      'g_ae_enc_{}_W'.format(l_i))))
            b_npy = np.load(os.path.join(os.path.join(raw_weights_dir,
                                                      'g_ae_enc_{}_b'.format(l_i))))
            alpha_npy = np.load(os.path.join(os.path.join(raw_weights_dir,
                                                      'g_ae_enc_prelu_{}_alpha'.format(l_i))))
            w_npy = np.squeeze(w_npy, axis=1)
            print('Loaded raw weights size: ', w_npy.shape)
            print('Loaded raw bias size: ', b_npy.shape)
            print('Loaded raw alphas size: ', alpha_npy.shape)
            # transpose weights 
            w_npy = w_npy.transpose(2, 1, 0)
            print('PyTorch layer alpha size: ', g_enc_l.act.weight.size())
            print('PyTorch layer weight size: ', g_enc_l.conv.weight.size())
            print('PyTorch layer bias size: ', g_enc_l.conv.bias.size())
            g_enc_l.act.weight.data = torch.FloatTensor(alpha_npy)
            g_enc_l.conv.weight.data = torch.FloatTensor(w_npy)
            g_enc_l.conv.bias.data = torch.FloatTensor(b_npy)
            print('Assigned weights and biases to layer')
            print('-' * 20)
        print('=' * 20)
        print('DECODER')
        for l_i, g_dec_l in enumerate(self.G.gen_dec):
            # read weights and biases
            w_npy = np.load(os.path.join(os.path.join(raw_weights_dir,
                                                      'g_ae_dec_{}_W'.format(l_i))))
            b_npy = np.load(os.path.join(os.path.join(raw_weights_dir,
                                                      'g_ae_dec_{}_b'.format(l_i))))
            if l_i < len(self.G.gen_dec) - 1:
                alpha_npy = np.load(os.path.join(os.path.join(raw_weights_dir,
                                                          'g_ae_dec_prelu_{}_alpha'.format(l_i))))
                print('Loaded raw alphas size: ', alpha_npy.shape)
            w_npy = np.squeeze(w_npy, axis=1)
            print('Loaded raw weights size: ', w_npy.shape)
            print('Loaded raw bias size: ', b_npy.shape)
            # transpose weights 
            w_npy = w_npy.transpose(2, 1, 0)
            if l_i < len(self.G.gen_dec) - 1:
                print('PyTorch layer alpha size: ', g_dec_l.act.weight.size())
                g_dec_l.act.weight.data = torch.FloatTensor(alpha_npy)
            print('PyTorch layer weight size: ', g_dec_l.conv.weight.size())
            print('PyTorch layer bias size: ', g_dec_l.conv.bias.size())
            g_dec_l.conv.weight.data = torch.FloatTensor(w_npy)
            g_dec_l.conv.bias.data = torch.FloatTensor(b_npy)
            print('Assigned weights and biases to layer')
            print('-' * 20)
        
    def generate(self, inwav, z = None):
        if self.z_dropout:
            self.G.apply(z_dropout)
        else:
            self.G.eval()
        #print('wave in size: ', inwav.size())
        #ori_len = inwav.size(2)
        #p_wav = make_divN(inwav.transpose(1, 2), 1024).transpose(1, 2)
        #print('p_wav size: ', p_wav.size())
        #print('ori_len: ', ori_len)
        #c_res, hall = self.infer_G(p_wav, z=z, ret_hid=True)
        #c_res = c_res[0, 0, :ori_len].data.numpy()
        #c_res = de_emphasize(c_res, self.preemph)
        #return c_res, hall
        N = 16384
        x = np.zeros((1, 1, N))
        c_res = None
        slice_idx = torch.zeros(1)
        for beg_i in range(0, inwav.shape[2], N):
            if inwav.shape[2] - beg_i < N:
                length = inwav.shape[2] - beg_i
                pad = N - length
            else:
                length = N
                pad = 0
            if pad  > 0:
                x[0, 0] = torch.cat((inwav[0, 0, beg_i:beg_i + length],
                                    torch.zeros(pad)), dim=0)
            else:
                x[0, 0] = inwav[0, 0, beg_i:beg_i + length]
            x = torch.FloatTensor(x)
            #canvas_w, hall = self.G(x, z=z, ret_hid=True)
            canvas_w, hall = self.infer_G(x, z=z, ret_hid=True, 
                                          slice_idx=slice_idx)
            nums = []
            for k in hall.keys():
                if 'enc' in k and 'zc' not in k:
                    nums.append(int(k.split('_')[1]))
            g_c = hall['enc_{}'.format(max(nums))]
            if z is None and hasattr(self.G, 'z'):
                # if z was created inside G as first inference
                z = self.G.z
            if pad > 0:
                canvas_w = canvas_w[0, 0, :-pad]
            canvas_w = canvas_w.data.numpy().squeeze()
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
            slice_idx += 1
        # de-emph
        c_res = de_emphasize(c_res, self.preemph)
        return c_res, g_c

    def discriminate(self, cwav, nwav):
        self.D.eval()
        d_in = torch.cat((cwav, nwav), dim=1)
        d_veredict, _ = self.D(d_in)
        return d_veredict

    def infer_G(self, nwav, cwav=None, z=None, ret_hid=False, slice_idx=0):
        if ret_hid:
            Genh, hall = self.G(nwav, z=z, ret_hid=ret_hid, slice_idx=slice_idx)
            return Genh, hall
        else:
            Genh = self.G(nwav, z=z, ret_hid=ret_hid, slice_idx=slice_idx)
            return Genh

    def infer_D(self, x_, ref):
        D_in = torch.cat((x_, ref), dim=1)
        return self.D(D_in)

    def gen_train_samples(self, clean_samples, noisy_samples, z_sample, 
                          iteration=None, slice_idx=0):
        if z_sample is not None:
            canvas_w = self.infer_G(noisy_samples, clean_samples, z=z_sample,
                                    slice_idx=slice_idx)
        else:
            canvas_w = self.infer_G(noisy_samples, clean_samples,
                                    slice_idx=slice_idx)
        sample_dif = noisy_samples - clean_samples
        # sample wavs
        for m in range(noisy_samples.size(0)):
            m_canvas = de_emphasize(canvas_w[m,
                                             0].cpu().data.numpy(),
                                    self.preemph)
            print('w{} max: {} min: {}'.format(m,
                                               m_canvas.max(),
                                               m_canvas.min()))
            wavfile.write(os.path.join(self.save_path,
                                       'sample_{}-'
                                       '{}.wav'.format(iteration,
                                                       m)),
                          int(16e3), m_canvas)
            m_clean = de_emphasize(clean_samples[m,
                                                 0].cpu().data.numpy(),
                                   self.preemph)
            m_noisy = de_emphasize(noisy_samples[m,
                                                 0].cpu().data.numpy(),
                                   self.preemph)
            m_dif = de_emphasize(sample_dif[m,
                                            0].cpu().data.numpy(),
                                 self.preemph)
            m_gtruth_path = os.path.join(self.save_path,
                                         'gtruth_{}.wav'.format(m))
            if not os.path.exists(m_gtruth_path):
                wavfile.write(os.path.join(self.save_path,
                                           'gtruth_{}.wav'.format(m)),
                              int(16e3), m_clean)
                wavfile.write(os.path.join(self.save_path,
                                           'noisy_{}.wav'.format(m)),
                              int(16e3), m_noisy)
                wavfile.write(os.path.join(self.save_path,
                                           'dif_{}.wav'.format(m)),
                              int(16e3), m_dif)


    def train(self, opts, dloader, criterion, l1_init, l1_dec_step,
              l1_dec_epoch, log_freq, va_dloader=None, smooth=0):

        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))

        """ Train the SEGAN """
        if opts.opt == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
            Dopt = optim.RMSprop(self.D.parameters(), lr=opts.d_lr)
        elif opts.opt == 'adam':
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr, betas=(0, 0.9))
            Dopt = optim.Adam(self.D.parameters(), lr=opts.d_lr, betas=(0, 0.9))
        else:
            raise ValueError('Unrecognized optimizer {}'.format(opts.opt))

        # attach opts to models so that they are saved altogether in ckpts
        self.G.optim = Gopt
        self.D.optim = Dopt
        
        # Build savers for end of epoch, storing up to 3 epochs each
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3,
                            optimizer=self.G.optim, prefix='EOE_G-')
        eoe_d_saver = Saver(self.D, opts.save_path, max_ckpts=3,
                            optimizer=self.D.optim, prefix='EOE_D-')
        num_batches = len(dloader) 
        l1_weight = l1_init
        iteration = 1
        timings = []
        evals = {}
        noisy_evals = {}
        noisy_samples = None
        clean_samples = None
        z_sample = None
        patience = opts.patience
        best_val_obj = 0
        # acumulator for exponential avg of valid curve
        acum_val_obj = 0
        alpha_val = opts.alpha_val
        # make label tensor
        label = torch.ones(opts.batch_size)
        if self.do_cuda:
            label = label.cuda()

        for epoch in range(1, opts.epoch + 1):
            beg_t = timeit.default_timer()
            self.G.train()
            self.D.train()
            for bidx, batch in enumerate(dloader, start=1):
                if epoch >= l1_dec_epoch:
                    if l1_weight > 0:
                        l1_weight -= l1_dec_step
                        # ensure it is 0 if it goes < 0
                        l1_weight = max(0, l1_weight)
                sample = batch
                if len(sample) == 4:
                    uttname, clean, noisy, slice_idx = batch
                else:
                    raise ValueError('Returned {} elements per '
                                     'sample?'.format(len(sample)))
                clean = clean.unsqueeze(1)
                noisy = noisy.unsqueeze(1)
                label.resize_(clean.size(0)).fill_(1)
                if self.do_cuda:
                    clean = clean.cuda()
                    noisy = noisy.cuda()
                if noisy_samples is None:
                    noisy_samples = noisy[:20, :, :].contiguous()
                    clean_samples = clean[:20, :, :].contiguous()
                # (1) D real update
                Dopt.zero_grad()
                total_d_fake_loss = 0
                total_d_real_loss = 0
                #Genh = self.G(noisy)
                Genh = self.infer_G(noisy, clean, slice_idx=slice_idx)
                lab = label
                #D_in = torch.cat((clean, noisy), dim=1)
                d_real, _ = self.infer_D(clean, noisy)
                #d_real, _ = self.D(D_in)
                d_real_loss = criterion(d_real.view(-1), lab)
                d_real_loss.backward()
                total_d_real_loss += d_real_loss
                
                # (2) D fake update
                #D_fake_in = torch.cat((Genh.detach(), noisy), dim=1)
                #d_fake, _ = self.D(D_fake_in)
                d_fake, _ = self.infer_D(Genh.detach(), noisy)
                # Make fake objective
                lab = label.fill_(0)
                d_fake_loss = criterion(d_fake.view(-1), lab)
                d_fake_loss.backward()
                total_d_fake_loss += d_fake_loss
                Dopt.step()

                d_loss = d_fake_loss + d_real_loss 

                # (3) G real update
                Gopt.zero_grad()
                lab = label.fill_(1)
                #d_fake_, _ = self.D(torch.cat((Genh, noisy), dim=1))
                d_fake_, _ = self.infer_D(Genh, noisy)
                g_adv_loss = criterion(d_fake_.view(-1), lab)
                g_l1_loss = l1_weight * F.l1_loss(Genh, clean)
                g_l2_reg = Genh.contiguous().view(-1).dot(Genh.contiguous().view(-1)) / \
                           Genh.contiguous().view(-1).size(0)
                g_l2_reg = self.canvas_l2 * g_l2_reg
                g_loss = g_adv_loss + g_l1_loss + g_l2_reg
                g_loss.backward()
                Gopt.step()
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if z_sample is None and not self.G.no_z:
                    # capture sample now that we know shape after first
                    # inference
                    z_sample = self.G.z[:20, :, :].contiguous()
                    print('z_sample size: ', z_sample.size())
                    if self.do_cuda:
                        z_sample = z_sample.cuda()
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    d_real_loss_v = d_real_loss.cpu().item()
                    d_fake_loss_v = d_fake_loss.cpu().item()
                    g_adv_loss_v = g_adv_loss.cpu().item()
                    g_l1_loss_v = g_l1_loss.cpu().item()
                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, ' \
                          'd_fake:{:.4f}, '.format(iteration, bidx,
                                                   len(dloader), epoch,
                                                   d_real_loss_v,
                                                   d_fake_loss_v)
                    log += 'g_adv:{:.4f}, g_l1:{:.4f} ' \
                           'l1_w: {:.2f}, canvas_l2: {:.4f} '\
                           'btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(g_adv_loss_v,
                                     g_l1_loss_v,
                                     l1_weight, 
                                     g_l2_reg,
                                     timings[-1],
                                     np.mean(timings))
                    print(log)
                    self.writer.add_scalar('D_real', d_real_loss_v,
                                           iteration)
                    self.writer.add_scalar('D_fake', d_fake_loss_v,
                                           iteration)
                    self.writer.add_scalar('G_adv', g_adv_loss_v,
                                           iteration)
                    self.writer.add_scalar('G_l1', g_l1_loss_v,
                                           iteration)
                    self.writer.add_scalar('G_canvas_l2', g_l2_reg,
                                           iteration)
                    self.writer.add_histogram('D_fake__hist', d_fake_.cpu().data,
                                              iteration, bins='sturges')
                    self.writer.add_histogram('D_fake_hist', d_fake.cpu().data,
                                              iteration, bins='sturges')
                    self.writer.add_histogram('D_real_hist', d_real.cpu().data,
                                              iteration, bins='sturges')
                    self.writer.add_histogram('Gz', Genh.cpu().data,
                                              iteration, bins='sturges')
                    self.writer.add_histogram('clean', clean.cpu().data,
                                              iteration, bins='sturges')
                    self.writer.add_histogram('noisy', noisy.cpu().data,
                                              iteration, bins='sturges')
                    # get D and G weights and plot their norms by layer and
                    # global
                    def model_weights_norm(model, total_name):
                        total_GW_norm = 0
                        for k, v in model.named_parameters():
                            if 'weight' in k:
                                W = v.data
                                W_norm = torch.norm(W)
                                self.writer.add_scalar('{}_Wnorm'.format(k),
                                                       W_norm,
                                                       iteration)
                                total_GW_norm += W_norm
                        self.writer.add_scalar('{}_Wnorm'.format(total_name),
                                               total_GW_norm,
                                               iteration)
                    model_weights_norm(self.G, 'Gtotal')
                    model_weights_norm(self.D, 'Dtotal')
                    if not opts.no_train_gen:
                        #canvas_w = self.G(noisy_samples, z=z_sample)
                        self.gen_train_samples(clean_samples, noisy_samples,
                                               z_sample,
                                               iteration=iteration, 
                                               slice_idx=slice_idx)
                iteration += 1

            if va_dloader is not None:
                if len(noisy_evals) == 0:
                    evals_, noisy_evals_ = self.evaluate(opts, va_dloader, 
                                                         log_freq, do_noisy=True)
                    for k, v in noisy_evals_.items():
                        if k not in noisy_evals:
                            noisy_evals[k] = []
                        noisy_evals[k] += v
                        self.writer.add_scalar('noisy-{}'.format(k), 
                                               noisy_evals[k][-1], epoch)
                else:
                    evals_ = self.evaluate(opts, va_dloader, 
                                           log_freq, do_noisy=False)
                for k, v in evals_.items():
                    if k not in evals:
                        evals[k] = []
                    evals[k] += v
                    self.writer.add_scalar('Genh-{}'.format(k), 
                                           evals[k][-1], epoch)
                val_obj = evals['covl'][-1] + evals['pesq'][-1]
                acum_val_obj = alpha_val * val_obj + \
                               (1 - alpha_val) * acum_val_obj
                self.writer.add_scalar('Genh-val_obj',
                                       val_obj, epoch)
                self.writer.add_scalar('Genh-SMOOTH_val_obj',
                                       acum_val_obj, epoch)
                if val_obj > best_val_obj:
                    # save models with true valid curve is minimum
                    self.G.save(self.save_path, iteration, True)
                    self.D.save(self.save_path, iteration, True)
                if acum_val_obj > best_val_obj:
                    print('Acum Val obj (COVL + SSNR) improved '
                          '{} -> {}'.format(best_val_obj,
                                            acum_val_obj))
                    best_val_obj = acum_val_obj
                    patience = opts.patience
                else:
                    patience -= 1
                    print('Val loss did not improve. Patience'
                          '{}/{}'.format(patience,
                                        opts.patience))
                    if patience <= 0:
                        print('STOPPING SEGAN TRAIN: OUT OF PATIENCE.')
                        break

            # save models in end of epoch with EOE savers
            self.G.save(self.save_path, iteration, saver=eoe_g_saver)
            self.D.save(self.save_path, iteration, saver=eoe_d_saver)


    def evaluate(self, opts, dloader, log_freq, do_noisy=False,
                 max_samples=1):
        """ Objective evaluation with PESQ and SSNR """
        self.G.eval()
        self.D.eval()
        evals = {'pesq':[], 'ssnr':[], 'csig':[],
                 'cbak':[], 'covl':[]}
        pesqs = []
        ssnrs = []
        if do_noisy:
            noisy_evals = {'pesq':[], 'ssnr':[], 'csig':[],
                           'cbak':[], 'covl':[]}
            npesqs = []
            nssnrs = []
        if not hasattr(self, 'pool'):
            self.pool = mp.Pool(opts.eval_workers)
        total_s = 0
        timings = []
        with torch.no_grad():
            # going over dataset ONCE
            for bidx, batch in enumerate(dloader, start=1):
                sample = batch
                if len(sample) == 4:
                    uttname, clean, noisy, slice_idx = batch
                else:
                    raise ValueError('Returned {} elements per '
                                     'sample?'.format(len(sample)))
                clean = clean
                noisy = noisy.unsqueeze(1)
                if self.do_cuda:
                    clean = clean.cuda()
                    noisy = noisy.cuda()
                #Genh = self.G(noisy).squeeze(1)
                Genh = self.infer_G(noisy, slice_idx=slice_idx).squeeze(1)
                clean_npy = clean.cpu().data.numpy()
                Genh_npy = Genh.cpu().data.numpy()
                clean_npy = np.apply_along_axis(de_emphasize, 0, clean_npy,
                                                self.preemph)
                Genh_npy = np.apply_along_axis(de_emphasize, 0, Genh_npy,
                                                self.preemph)
                beg_t = timeit.default_timer()
                if do_noisy:
                    noisy_npy = noisy.cpu().data.numpy()
                    noisy_npy = np.apply_along_axis(de_emphasize, 0, noisy_npy,
                                                    self.preemph)
                    args = [(clean_npy[i], Genh_npy[i], noisy_npy[i]) for i in \
                            range(clean.size(0))]
                else:
                    args = [(clean_npy[i], Genh_npy[i], None) for i in \
                            range(clean.size(0))]
                map_ret = self.pool.map(composite_helper, args)
                end_t = timeit.default_timer()
                print('Time to process eval with {} samples ' \
                      ': {} s'.format(clean.size(0), end_t - beg_t))
                if bidx >= max_samples:
                    break

            def fill_ret_dict(ret_dict, in_dict):
                for k, v in in_dict.items():
                    ret_dict[k].append(v)

            if do_noisy:
                for eval_, noisy_eval_ in map_ret:
                    fill_ret_dict(evals, eval_)
                    fill_ret_dict(noisy_evals, noisy_eval_)
                return evals, noisy_evals
            else:
                for eval_ in map_ret:
                    fill_ret_dict(evals, eval_)
                return evals

class WSEGAN(SEGAN):

    def __init__(self, opts, name='WSEGAN',
                 generator=None,
                 discriminator=None):
        self.lbd = 1
        self.critic_iters = 1
        if hasattr(opts, 'misalign_pair'):
            self.misalign_pair = opts.misalign_pair
        else:
            self.misalign_pair = False
        if hasattr(opts, 'interf_pair'):
            self.interf_pair = opts.interf_pair
        else:
            self.interf_pair = False
        if hasattr(opts, 'fake_sines'):
            self.fake_sines = opts.fake_sines
        else:
            self.fake_sines = False
        if hasattr(opts, 'pow_weight'):
            self.pow_weight = opts.pow_weight
        else:
            self.pow_weight = 10000
        if hasattr(opts, 'vanilla_gan'):
            self.vanilla_gan = opts.vanilla_gan
        else:
            self.vanilla_gan = False
        if hasattr(opts, 'n_fft'):
            self.n_fft = opts.n_fft
        else:
            self.n_fft = 2048
        G = None
        self.g_enc_fmaps = opts.g_enc_fmaps
        if hasattr(opts, 'nigenerator') and opts.nigenerator:
            if opts.g_act == 'prelu':
                self.g_enc_act = [nn.PReLU(fmaps) for fmaps in self.g_enc_fmaps]
            elif opts.g_act == 'tanh':
                self.g_enc_act = 'Tanh'
                self.g_dec_act = None
            elif opts.g_act == 'relu':
                self.g_enc_act = 'ReLU'
                self.g_dec_act = None
            else:
                raise TypeError('Unrecognized G activation: ', opts.g_act)
            # Build G and D
            G = ARGenerator(1, opts.g_enc_fmaps,
                            opts.kwidth,
                            self.g_enc_act,
                            pooling=opts.pooling_size,
                            z_dim=self.g_enc_fmaps[-1],
                            cuda=opts.cuda,
                            bias=opts.bias)

        if hasattr(opts, 'ardiscriminator') and opts.ardiscriminator:
            discriminator = ARDiscriminator(fmaps=[256] * 5)
        super(WSEGAN, self).__init__(opts, name, 
                                     G, discriminator)
        self.G.apply(wsegan_weights_init)
        self.D.apply(wsegan_weights_init)
        self.l1_weight = opts.l1_weight

        self.num_devices = 1
        if opts.cuda:
            self.num_devices = torch.cuda.device_count()
            if self.num_devices > 1:
                print('Using {:1d} GPUs for G and D'
                      ''.format(torch.cuda.device_count()))
                self.G = nn.DataParallel(self.G)
                self.D = nn.DataParallel(self.D)

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        bsz = real_data.size(0)
        #alpha = torch.rand(bsz, 1)
        # regularize real data
        alpha = torch.ones(bsz, 1)
        alpha = alpha.expand(bsz, real_data.nelement() // bsz).contiguous()
        alpha = alpha.view(real_data.size())
        alpha = alpha.to('cuda') if self.do_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if self.do_cuda:
            interpolates = interpolates.to('cuda')

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)[0][:, :1]

        grad_out = torch.ones(disc_interpolates.size())

        if self.do_cuda:
            grad_out = grad_out.cuda()
        
        gradients = autograd.grad(outputs=disc_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=grad_out,
                                  create_graph=True, retain_graph=True,
                                  only_inputs=True)[0]
        gr = gradients.view(gradients.size(0), -1)
        #gradient_p = torch.mean((1. - torch.sqrt(1e-8 + \
        #                                         torch.sum(gr ** 2, \
        #                                                   dim=1))) ** 2)
                                 
        gradient_p = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lbd
        return gradient_p

    def sample_dloader(self, dloader):
        sample = next(dloader.__iter__())
        batch = sample
        uttname, clean, noisy, slice_idx = batch
        clean = clean.unsqueeze(1)
        noisy = noisy.unsqueeze(1)
        if self.do_cuda:
            clean = clean.to('cuda')
            noisy = noisy.to('cuda')
            slice_idx = slice_idx.to('cuda')
        return uttname, clean, noisy, slice_idx

    def infer_G(self, nwav, cwav=None, z=None, ret_hid=False, slice_idx=0,
                att_weight=0):
        Genh = self.G(nwav, z=z, ret_hid=ret_hid, slice_idx=slice_idx,
                      att_weight=att_weight)
        return Genh

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step,
              l1_dec_epoch, log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))
        if opts.opt == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
            Dopt = optim.RMSprop(self.D.parameters(), lr=opts.d_lr)
            #hfDopt = optim.RMSprop(self.hf_D.parameters(), lr=opts.d_lr)
        elif opts.opt == 'adam':
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr, betas=(0.5,
                                                                        0.9))
            Dopt = optim.Adam(self.D.parameters(), lr=opts.d_lr, betas=(0.5,
                                                                        0.9))
            #hfDopt = optim.Adam(self.hfD.parameters(), lr=opts.d_lr,
            #                    betas=(0.5, 0.9))
        else:
            raise ValueError('Unrecognized optimizer {}'.format(opts.opt))

        # attach opts to models so that they are saved altogether in ckpts
        self.G.optim = Gopt
        self.D.optim = Dopt
        #self.hfD.optim = hfDopt
        
        # Build savers for end of epoch, storing up to 3 epochs each
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3,
                            optimizer=self.G.optim, prefix='EOE_G-')
        eoe_d_saver = Saver(self.D, opts.save_path, max_ckpts=3,
                            optimizer=self.D.optim, prefix='EOE_D-')
        num_batches = len(dloader) 
        l1_weight = l1_init
        iteration = 1
        timings = []
        evals = {}
        noisy_evals = {}
        noisy_samples = None
        clean_samples = None
        z_sample = None
        patience = opts.patience
        best_val_obj = np.inf
        # acumulator for exponential avg of valid curve
        acum_val_obj = 0
        alpha_val = opts.alpha_val
        # begin with att weight at zero, and increase after epoch 2
        att_weight = 0
        att_w_inc = 0.2
        G = self.G
        D = self.D
        if self.num_devices > 1:
            G = self.G.module
            D = self.G.module

        for iteration in range(1, opts.epoch * len(dloader) + 1):
            beg_t = timeit.default_timer()
            if iteration >= 3 * len(dloader) and \
               (iteration % len(dloader) == 0):
                att_weight += att_w_inc
            uttname, clean, noisy, slice_idx = self.sample_dloader(dloader)
            bsz = clean.size(0)
            # grads
            Dopt.zero_grad()
            D_in = torch.cat((clean, noisy), dim=1)
            d_real, _ = self.infer_D(clean, noisy)
            rl_lab = torch.ones(d_real.size()).cuda()
            #d_real_loss = d_real.mean()
            if self.vanilla_gan:
                cost = F.binary_cross_entropy_with_logits
            else:
                cost = F.mse_loss
            d_real_loss = cost(d_real, rl_lab)
            Genh = self.infer_G(noisy, clean, slice_idx=slice_idx,
                                att_weight=att_weight)
            fake = Genh.detach()
            d_fake, _ = self.infer_D(fake, noisy)
            fk_lab = torch.zeros(d_fake.size()).cuda()
            
            d_fake_loss = cost(d_fake, fk_lab)

            #gradient_penalty = self.calc_gradient_penalty(self.D,
            #                                              D_in.data,
            #                                              fake.data)
            d_weight = 0.5 # count only d_fake and d_real
            d_loss = d_fake_loss + d_real_loss
            if self.misalign_pair:
                clean_shuf = list(torch.chunk(clean, clean.size(0), dim=0))
                shuffle(clean_shuf)
                clean_shuf = torch.cat(clean_shuf, dim=0)
                d_fake_shuf, _ = self.infer_D(clean, clean_shuf)
                d_fake_shuf_loss = cost(d_fake_shuf, fk_lab)
                d_weight = 1 / 3 # count 3 components now
                d_loss + d_fake_shuf_loss
                #d_loss = (1/3) * d_fake_loss + (1/3) * d_real_loss + \
                #         (1/3) * d_fake_shuf_loss
            if self.interf_pair:
                # put interferring squared signals with random amplitude and
                # freq as fake signals mixed with clean data
                freqs = [250, 1000, 4000]
                amps = [0.01, 0.05, 0.1, 1]
                bsz = clean.size(0)
                squares = []
                t = np.linspace(0, 2, 32000)
                for _ in range(bsz):
                    f_ = random.choice(freqs)
                    a_ = random.choice(amps)
                    sq = a_ * signal.square(2 * np.pi * f_ * t)
                    sq = sq[:clean.size(-1)].reshape((1, -1))
                    squares.append(torch.FloatTensor(sq))
                squares = torch.cat(squares, dim=0).unsqueeze(1)
                if clean.is_cuda:
                    squares = squares.to('cuda')
                interf = clean + squares
                d_fake_inter, _ = self.infer_D(interf, noisy)
                d_fake_inter_loss = cost(d_fake_inter, fk_lab)
                d_weight = 1 / 4 # count 4 components in d loss now
                d_loss += d_fake_inter_loss

            d_loss = d_weight * d_loss

            d_loss.backward()
            #wasserstein_D = d_real_loss - d_fake_loss
            #wasserstein_D = torch.zeros(1)
            Dopt.step()

            Gopt.zero_grad()
            #Genh = self.infer_G(noisy, clean, slice_idx=slice_idx)
            d_fake_, _ = self.infer_D(Genh, noisy)
            g_adv_loss = cost(d_fake_, torch.ones(d_fake_.size()).cuda())

            # POWER Loss -----------------------------------
            # make stft of gtruth
            clean_stft = torch.stft(clean.squeeze(1), 
                                    n_fft=min(clean.size(-1), self.n_fft), 
                                    hop_length=160,
                                    win_length=320,
                                    normalized=True)
            clean_mod = torch.norm(clean_stft, 2, dim=3)
            #clean_mod_pow = clean_mod ** 2
            clean_mod_pow = 10 * torch.log10(clean_mod ** 2 + 10e-20)
            Genh_stft = torch.stft(Genh.squeeze(1), 
                                   n_fft=min(Genh.size(-1), self.n_fft),
                                   hop_length=160, 
                                   win_length=320, normalized=True)
            Genh_mod = torch.norm(Genh_stft, 2, dim=3)
            #Genh_mod_pow = Genh_mod ** 2
            Genh_mod_pow = 10 * torch.log10(Genh_mod ** 2 + 10e-20)
            #pow_loss = self.pow_weight * F.mse_loss(Genh_mod_pow, clean_mod_pow)
            pow_loss = self.pow_weight * F.l1_loss(Genh_mod_pow, clean_mod_pow)
            #pow_loss.backward()
            #G_cost = g_adv_loss + g_adv_hf_loss + pow_loss
            G_cost = g_adv_loss + pow_loss
            if self.l1_weight > 0:
                # look for additive files to build batch mask
                mask = torch.zeros(bsz, 1, Genh.size(2))
                if opts.cuda:
                    mask = mask.to('cuda')
                for utt_i, uttn in enumerate(uttname):
                    if 'additive' in uttn:
                        mask[utt_i, 0, :] = 1.
                den_loss = self.l1_weight * F.l1_loss(Genh * mask,
                                                      clean * mask)
                G_cost += den_loss
            else:
                den_loss = torch.zeros(1)
            G_cost.backward()
            Gopt.step()
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if noisy_samples is None:
                noisy_samples = noisy[:20, :, :].contiguous()
                clean_samples = clean[:20, :, :].contiguous()
            if z_sample is None and not G.no_z:
                # capture sample now that we know shape after first
                # inference
                z_sample = G.z[:20, :, :].contiguous()
                print('z_sample size: ', z_sample.size())
                if self.do_cuda:
                    z_sample = z_sample.cuda()
            if iteration % log_freq == 0:
                log = 'Iter {}/{} ({} bpe) d_loss:{:.4f}, ' \
                      'g_loss: {:.4f}, pow_loss: {:.4f}, ' \
                      'den_loss: {:.4f} ' \
                      ''.format(iteration,
                                len(dloader) * opts.epoch,
                                len(dloader),
                                d_loss.item(),
                                G_cost.item(),
                                pow_loss.item(),
                                den_loss.item())

                log += 'btime: {:.4f} s, mbtime: {:.4f} s' \
                       ''.format(timings[-1],
                                 np.mean(timings))
                print(log)
                self.writer.add_scalar('D_loss', d_loss.item(),
                                       iteration)
                self.writer.add_scalar('G_loss', G_cost.item(),
                                       iteration)
                self.writer.add_scalar('G_adv_loss', g_adv_loss.item(),
                                       iteration)
                self.writer.add_scalar('att_weight', att_weight,
                                       iteration)
                self.writer.add_scalar('G_pow_loss', pow_loss.item(),
                                       iteration)
                self.writer.add_histogram('clean_mod_pow',
                                          clean_mod_pow.cpu().data,
                                          iteration,
                                          bins='sturges')
                self.writer.add_histogram('Genh_mod_pow',
                                          Genh_mod_pow.cpu().data,
                                          iteration,
                                          bins='sturges')
                self.writer.add_histogram('Gz', Genh.cpu().data,
                                          iteration, bins='sturges')
                self.writer.add_histogram('clean', clean.cpu().data,
                                          iteration, bins='sturges')
                self.writer.add_histogram('noisy', noisy.cpu().data,
                                          iteration, bins='sturges')
                if hasattr(G, 'skips'):
                    for skip_id, alpha in G.skips.items():
                        skip = alpha['alpha']
                        if skip.skip_type == 'alpha':
                            self.writer.add_histogram('skip_alpha_{}'.format(skip_id),
                                                      skip.skip_k.data,
                                                      iteration, 
                                                      bins='sturges')
                if self.linterp:
                    for dec_i, gen_dec in enumerate(G.gen_dec, start=1):
                        if not hasattr(gen_dec, 'linterp_aff'):
                            continue
                        linterp_w = gen_dec.linterp_aff.linterp_w
                        linterp_b = gen_dec.linterp_aff.linterp_b
                        self.writer.add_histogram('linterp_w_{}'.format(dec_i),
                                                  linterp_w.data,
                                                  iteration, bins='sturges')

                        self.writer.add_histogram('linterp_b_{}'.format(dec_i),
                                                  linterp_b.data,
                                                  iteration, bins='sturges')

                # get D and G weights and plot their norms by layer and global
                def model_weights_norm(model, total_name):
                    total_GW_norm = 0
                    for k, v in model.named_parameters():
                        if 'weight' in k:
                            W = v.data
                            W_norm = torch.norm(W)
                            self.writer.add_scalar('{}_Wnorm'.format(k),
                                                   W_norm,
                                                   iteration)
                            total_GW_norm += W_norm
                    self.writer.add_scalar('{}_Wnorm'.format(total_name),
                                           total_GW_norm,
                                           iteration)
                model_weights_norm(G, 'Gtotal')
                model_weights_norm(D, 'Dtotal')
                if not opts.no_train_gen:
                    #canvas_w = self.G(noisy_samples, z=z_sample)
                    self.gen_train_samples(clean_samples, noisy_samples,
                                           z_sample,
                                           iteration=iteration, 
                                           slice_idx=slice_idx)
                if va_dloader is not None:
                    if len(noisy_evals) == 0:
                        sd, nsd = self.evaluate(opts, va_dloader,
                                                log_freq, do_noisy=True)
                        self.writer.add_scalar('noisy_SD',
                                               nsd, iteration)
                    else:
                        sd = self.evaluate(opts, va_dloader, 
                                           log_freq, do_noisy=False)
                    self.writer.add_scalar('Genh_SD',
                                           sd, iteration)
                    print('Eval SD: {:.3f} dB, NSD: {:.3f} dB'.format(sd, nsd))
                    if sd < best_val_obj:
                        self.G.save(self.save_path, iteration, True)
                        self.D.save(self.save_path, iteration, True)
                        best_val_obj = sd
            if iteration % len(dloader) == 0:
                # save models in end of epoch with EOE savers
                G.save(self.save_path, iteration, saver=eoe_g_saver)
                D.save(self.save_path, iteration, saver=eoe_d_saver)

    def generate(self, inwav, z = None):
        if self.z_dropout:
            self.G.apply(z_dropout)
        else:
            self.G.eval()
        print('wave in size: ', inwav.size())
        ori_len = inwav.size(2)
        p_wav = make_divN(inwav.transpose(1, 2), 1024).transpose(1, 2)
        print('p_wav size: ', p_wav.size())
        print('ori_len: ', ori_len)
        c_res, hall = self.infer_G(p_wav, z=z, ret_hid=True)
        print('c_res size: ', c_res.size())
        c_res = c_res[0, 0, :ori_len].cpu().data.numpy()
        print('c_res after trim: ', c_res.shape)
        c_res = de_emphasize(c_res, self.preemph)
        return c_res, hall

    def evaluate(self, opts, dloader, log_freq, do_noisy=False,
                 max_samples=1):
        """ Objective evaluation with MCD """
        self.G.eval()
        timings = []
        sds = []
        if do_noisy:
            nsds = []
        with torch.no_grad():
            # going over dataset ONCE
            for bidx, batch in enumerate(dloader, start=1):
                sample = batch
                if len(sample) == 4:
                    uttname, clean, noisy, slice_idx = batch
                else:
                    raise ValueError('Returned {} elements per '
                                     'sample?'.format(len(sample)))
                clean = clean
                noisy = noisy.unsqueeze(1)
                if self.do_cuda:
                    clean = clean.cuda()
                    noisy = noisy.cuda()
                Genh = self.infer_G(noisy, slice_idx=slice_idx).squeeze(1)
                clean_stft = torch.stft(clean.squeeze(1), 
                                        n_fft=min(clean.size(-1), self.n_fft), 
                                        hop_length=160,
                                        win_length=320,
                                        normalized=True)
                clean_mod = torch.norm(clean_stft, 2, dim=3)
                clean_mod_pow = 10 * torch.log10(clean_mod ** 2 + 10e-20)
                Genh_stft = torch.stft(Genh.detach().squeeze(1), 
                                       n_fft=min(Genh.size(-1), self.n_fft),
                                       hop_length=160, 
                                       win_length=320, normalized=True)
                Genh_mod = torch.norm(Genh_stft, 2, dim=3)
                Genh_mod_pow = 10 * torch.log10(Genh_mod ** 2 + 10e-20)
                spec_dis_db = F.l1_loss(Genh_mod_pow, clean_mod_pow)
                sds.append(spec_dis_db)
                if do_noisy:
                    noisy_stft = torch.stft(noisy.detach().squeeze(1), 
                                           n_fft=min(noisy.size(-1), self.n_fft),
                                           hop_length=160, 
                                           win_length=320, normalized=True)
                    noisy_mod = torch.norm(noisy_stft, 2, dim=3)
                    noisy_mod_pow = 10 * torch.log10(noisy_mod ** 2 + 10e-20)
                    nspec_dis_db = F.l1_loss(noisy_mod_pow, clean_mod_pow)
                    nsds.append(nspec_dis_db)
            if do_noisy:
                return np.mean(sds), np.mean(nsds)
            return np.mean(sds)

class AEWSEGAN(WSEGAN):

    def __init__(self, opts, name='AEWSEGAN',
                 generator=None,
                 discriminator=None):
        if hasattr(opts, 'l1_loss'):
            self.l1_loss = opts.l1_loss
        else:
            self.l1_loss = False
        super().__init__(opts, name=name, generator=generator,
                         discriminator=discriminator)
        # delete discriminator
        self.D = None

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step,
              l1_dec_epoch, log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))
        if opts.opt == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
        elif opts.opt == 'adam':
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr, betas=(0.5,
                                                                        0.9))
        else:
            raise ValueError('Unrecognized optimizer {}'.format(opts.opt))

        # attach opts to models so that they are saved altogether in ckpts
        self.G.optim = Gopt
        
        # Build savers for end of epoch, storing up to 3 epochs each
        eoe_g_saver = Saver(self.G, opts.save_path, max_ckpts=3,
                            optimizer=self.G.optim, prefix='EOE_G-')
        num_batches = len(dloader) 
        l2_weight = l1_init
        iteration = 1
        timings = []
        evals = {}
        noisy_evals = {}
        noisy_samples = None
        clean_samples = None
        z_sample = None
        patience = opts.patience
        best_val_obj = np.inf
        # acumulator for exponential avg of valid curve
        acum_val_obj = 0
        alpha_val = opts.alpha_val
        G = self.G

        for iteration in range(1, opts.epoch * len(dloader) + 1):
            beg_t = timeit.default_timer()
            uttname, clean, noisy, slice_idx = self.sample_dloader(dloader)
            bsz = clean.size(0)
            Genh = self.infer_G(noisy, clean, slice_idx=slice_idx,
                                att_weight=0)

            Gopt.zero_grad()
            if self.l1_loss:
                loss = F.l1_loss(Genh, clean)
            else:
                loss = F.mse_loss(Genh, clean)
            loss.backward()
            Gopt.step()
            end_t = timeit.default_timer()
            timings.append(end_t - beg_t)
            beg_t = timeit.default_timer()
            if noisy_samples is None:
                noisy_samples = noisy[:20, :, :].contiguous()
                clean_samples = clean[:20, :, :].contiguous()
            if z_sample is None and not G.no_z:
                # capture sample now that we know shape after first
                # inference
                z_sample = G.z[:20, :, :].contiguous()
                print('z_sample size: ', z_sample.size())
                if self.do_cuda:
                    z_sample = z_sample.cuda()
            if iteration % log_freq == 0:
                # POWER Loss (not used to backward) -----------------------------------
                # make stft of gtruth
                clean_stft = torch.stft(clean.squeeze(1), 
                                        n_fft=min(clean.size(-1), self.n_fft), 
                                        hop_length=160,
                                        win_length=320,
                                        normalized=True)
                clean_mod = torch.norm(clean_stft, 2, dim=3)
                clean_mod_pow = 10 * torch.log10(clean_mod ** 2 + 10e-20)
                Genh_stft = torch.stft(Genh.detach().squeeze(1), 
                                       n_fft=min(Genh.size(-1), self.n_fft),
                                       hop_length=160, 
                                       win_length=320, normalized=True)
                Genh_mod = torch.norm(Genh_stft, 2, dim=3)
                Genh_mod_pow = 10 * torch.log10(Genh_mod ** 2 + 10e-20)
                pow_loss = F.l1_loss(Genh_mod_pow, clean_mod_pow)
                log = 'Iter {}/{} ({} bpe) g_l2_loss:{:.4f}, ' \
                      'pow_loss: {:.4f}, ' \
                      ''.format(iteration,
                                len(dloader) * opts.epoch,
                                len(dloader),
                                loss.item(),
                                pow_loss.item())

                log += 'btime: {:.4f} s, mbtime: {:.4f} s' \
                       ''.format(timings[-1],
                                 np.mean(timings))
                print(log)
                self.writer.add_scalar('g_l2_loss', loss.item(),
                                       iteration)
                self.writer.add_scalar('G_pow_loss', pow_loss.item(),
                                       iteration)
                self.writer.add_histogram('clean_mod_pow',
                                          clean_mod_pow.cpu().data,
                                          iteration,
                                          bins='sturges')
                self.writer.add_histogram('Genh_mod_pow',
                                          Genh_mod_pow.cpu().data,
                                          iteration,
                                          bins='sturges')
                self.writer.add_histogram('Gz', Genh.cpu().data,
                                          iteration, bins='sturges')
                self.writer.add_histogram('clean', clean.cpu().data,
                                          iteration, bins='sturges')
                self.writer.add_histogram('noisy', noisy.cpu().data,
                                          iteration, bins='sturges')
                if hasattr(G, 'skips'):
                    for skip_id, alpha in G.skips.items():
                        skip = alpha['alpha']
                        if skip.skip_type == 'alpha':
                            self.writer.add_histogram('skip_alpha_{}'.format(skip_id),
                                                      skip.skip_k.data,
                                                      iteration, 
                                                      bins='sturges')
                # get D and G weights and plot their norms by layer and global
                def model_weights_norm(model, total_name):
                    total_GW_norm = 0
                    for k, v in model.named_parameters():
                        if 'weight' in k:
                            W = v.data
                            W_norm = torch.norm(W)
                            self.writer.add_scalar('{}_Wnorm'.format(k),
                                                   W_norm,
                                                   iteration)
                            total_GW_norm += W_norm
                    self.writer.add_scalar('{}_Wnorm'.format(total_name),
                                           total_GW_norm,
                                           iteration)
                #model_weights_norm(G, 'Gtotal')
                #model_weights_norm(D, 'Dtotal')
                if not opts.no_train_gen:
                    #canvas_w = self.G(noisy_samples, z=z_sample)
                    self.gen_train_samples(clean_samples, noisy_samples,
                                           z_sample,
                                           iteration=iteration, 
                                           slice_idx=slice_idx)
                if va_dloader is not None:
                    if len(noisy_evals) == 0:
                        sd, nsd = self.evaluate(opts, va_dloader,
                                                log_freq, do_noisy=True)
                        self.writer.add_scalar('noisy_SD',
                                               nsd, iteration)
                    else:
                        sd = self.evaluate(opts, va_dloader, 
                                           log_freq, do_noisy=False)
                    self.writer.add_scalar('Genh_SD',
                                           sd, iteration)
                    print('Eval SD: {:.3f} dB, NSD: {:.3f} dB'.format(sd, nsd))
                    if sd < best_val_obj:
                        self.G.save(self.save_path, iteration, True)
                        best_val_obj = sd
            if iteration % len(dloader) == 0:
                # save models in end of epoch with EOE savers
                G.save(self.save_path, iteration, saver=eoe_g_saver)
                """
                val_obj = evals['covl'][-1] + evals['pesq'][-1]
                acum_val_obj = alpha_val * val_obj + \
                               (1 - alpha_val) * acum_val_obj
                self.writer.add_scalar('Genh-val_obj',
                                       val_obj, epoch)
                self.writer.add_scalar('Genh-SMOOTH_val_obj',
                                       acum_val_obj, epoch)
                if val_obj > best_val_obj:
                    # save models with true valid curve is minimum
                    G.save(self.save_path, iteration, True)
                    D.save(self.save_path, iteration, True)
                if acum_val_obj > best_val_obj:
                    print('Acum Val obj (COVL + SSNR) improved '
                          '{} -> {}'.format(best_val_obj,
                                            acum_val_obj))
                    best_val_obj = acum_val_obj
                    patience = opts.patience
                else:
                    patience -= 1
                    print('Val loss did not improve. Patience'
                          '{}/{}'.format(patience,
                                        opts.patience))
                    if patience <= 0:
                        print('STOPPING WSEGAN TRAIN: OUT OF PATIENCE.')
                        break
                """

