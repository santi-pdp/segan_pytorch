import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from ..datasets import *
from ..utils import *
from scipy.io import wavfile
import numpy as np
import timeit
import random
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from .generator import *
from .discriminator import *
from .core import *
import json
import os


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print('Initialzing weight to 0.0, 0.01')
        m.weight.data.normal_(0.0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            print('bias to 0 for module: ', m)
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)

class SEGAN(Model):

    def __init__(self, opts, name='SEGAN'):
        super(SEGAN, self).__init__(name)
        self.opts = opts
        self.pesq_objective = opts.pesq_objective
        self.preemph = opts.preemph
        self.save_path = opts.save_path
        self.do_cuda = opts.cuda
        self.g_dropout = opts.g_dropout
        self.z_dim = opts.z_dim
        self.g_enc_fmaps = opts.g_enc_fmaps
        self.g_bias = opts.g_bias
        self.d_iter = opts.d_iter
        self.max_ma = opts.max_ma
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

        # Build G and D
        self.G = Generator(1, self.g_enc_fmaps, opts.kwidth,
                           self.g_enc_act,
                           opts.g_bnorm, opts.g_dropout, 
                           pooling=2, z_dim=opts.z_dim,
                           z_all=opts.z_all,
                           cuda=opts.cuda,
                           skip=opts.skip,
                           skip_blacklist=opts.skip_blacklist,
                           dec_activations=self.g_dec_act,
                           bias=opts.g_bias,
                           aal=opts.g_aal,
                           wd=opts.wd,
                           core2d=opts.core2d,
                           core2d_kwidth=opts.core2d_kwidth)
        if not opts.no_winit:
            self.G.apply(weights_init)
        print('Generator: ', self.G)

        self.d_enc_fmaps = opts.d_enc_fmaps
        self.D = Discriminator(2, self.d_enc_fmaps, opts.kwidth,
                               nn.LeakyReLU(0.3), bnorm=opts.d_bnorm,
                               pooling=2, SND=opts.SND,
                               rnn_pool=opts.D_rnn_pool,
                               dropout=opts.d_dropout,
                               rnn_size=opts.D_rnn_size)
        if not opts.no_winit:
            self.D.apply(weights_init)
        print('Discriminator: ', self.D)
        if self.do_cuda:
            self.D.cuda()
            self.G.cuda()
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))

    def train(self, opts, dloader, criterion, l1_init, l1_dec_step,
              l1_dec_epoch, log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        Gopt = getattr(optim, opts.g_opt)(self.G.parameters(), lr=opts.g_lr)
                                          #betas=(opts.beta_1, 0.99))
        Dopt = getattr(optim, opts.d_opt)(self.D.parameters(), lr=opts.d_lr)
                                          #betas=(opts.beta_1, 0.99))

        num_batches = len(dloader) 

        #self.load_weights()

        self.G.train()
        self.D.train()

        l1_weight = l1_init
        global_step = 1
        timings = []
        noisy_samples = None
        clean_samples = None
        z_sample = None
        for epoch in range(1, opts.epoch + 1):
            beg_t = timeit.default_timer()
            for bidx, batch in enumerate(dloader, start=1):
                if epoch >= l1_dec_epoch:
                    if l1_weight > 0:
                        l1_weight -= l1_dec_step
                        # ensure it is 0 if it goes < 0
                        l1_weight = max(0, l1_weight)
                sample = batch
                if len(sample) == 2:
                    clean, noisy = batch
                elif len(sample) == 3:
                    clean, noisy, pesq = batch
                else:
                    clean, noisy, pesq, ssnr = batch
                clean = Variable(clean.unsqueeze(1))
                noisy = Variable(noisy.unsqueeze(1))
                d_weight = 1./3
                if self.pesq_objective:
                    d_weight = 1./2
                    lab = Variable(pesq).view(-1, 1)
                    #lab = Variable(4.5 * torch.ones(pesq.size(0), 1))
                else:
                    lab = Variable((1 - smooth) * torch.ones(clean.size(0), 1))
                if self.do_cuda:
                    clean = clean.cuda()
                    noisy = noisy.cuda()
                    lab = lab.cuda()
                if noisy_samples is None:
                    noisy_samples = noisy[:20, :, :]
                    clean_samples = clean[:20, :, :]
                # (1) D real update
                Dopt.zero_grad()
                D_in = torch.cat((clean, noisy), dim=1)
                d_real = self.D(D_in)
                d_real_loss = d_weight * criterion(d_real, lab)
                d_real_loss.backward()
                
                # (2) D fake update
                Genh = self.G(noisy)
                if self.pesq_objective:
                    D_fake_in = torch.cat((Genh.detach(), clean), dim=1)
                else:
                    D_fake_in = torch.cat((Genh.detach(), noisy), dim=1)
                d_fake = self.D(D_fake_in)
                if not self.pesq_objective:
                    # regular fake objective
                    lab.data.fill_(0)
                d_fake_loss = d_weight * criterion(d_fake, lab)
                d_fake_loss.backward()
                Dopt.step()

                d_loss = d_fake_loss  + d_real_loss

                # (2.1) In case we dont have PESQ objective
                if not self.pesq_objective:
                    slice_idx = np.random.randint(1, min(clean.size(2),
                                                         self.max_ma))
                    # create mis-alignment fake signal
                    ma_clean1 = clean[:, :, :slice_idx]
                    ma_clean2 = clean[:, :, slice_idx:]
                    ma_clean = torch.cat((ma_clean2, ma_clean1), dim=2)
                    ma_clean = ma_clean.contiguous()
                    D_fakema_in = torch.cat((ma_clean, clean), dim=1)
                    d_ma_fake = self.D(D_fakema_in)
                    d_ma_fake_loss = d_weight * criterion(d_ma_fake, lab)
                    d_ma_fake_loss.backward()
                    Dopt.step()
                    d_loss = d_loss + d_ma_fake_loss


                # (3) G real update
                Gopt.zero_grad()
                if self.pesq_objective:
                    g_weight = 1./2
                    lab = Variable(pesq).view(-1, 1)
                    if self.do_cuda:
                        lab = lab.cuda()
                else:
                    g_weight = 1
                    lab.data.fill_(1 - smooth)
                d_fake_ = self.D(torch.cat((Genh, noisy), dim=1))
                g_adv_loss = g_weight * criterion(d_fake_, lab)
                #g_adv_loss.backward()
                g_l1_loss = l1_weight * F.l1_loss(Genh, clean)
                #g_l1_loss.backward()
                g_loss = g_adv_loss + g_l1_loss
                if self.pesq_objective:
                    lab.data.fill_(4.5)
                    # (3.1) Additional step to match clean, clean PESQ
                    d_fake2_ = self.D(torch.cat((Genh, clean), dim=1))
                    g_adv2_loss = g_weight * criterion(d_fake2_, lab)
                    #g_adv2_loss.backward()
                    g_loss += g_adv2_loss
                g_loss.backward()
                Gopt.step()
                if z_sample is None:
                    # capture sample now that we know shape after first
                    # inference
                    z_sample = self.G.z
                    print('z_sample size: ', z_sample.size())
                    if self.do_cuda:
                        z_sample = z_sample.cuda()
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    d_real_loss_v = np.asscalar(d_real_loss.cpu().data.numpy())
                    d_fake_loss_v = np.asscalar(d_fake_loss.cpu().data.numpy())
                    if not self.pesq_objective:
                        d_ma_fake_loss_v = np.asscalar(d_ma_fake_loss.cpu().data.numpy())
                    g_adv_loss_v = np.asscalar(g_adv_loss.cpu().data.numpy())
                    g_l1_loss_v = np.asscalar(g_l1_loss.cpu().data.numpy())
                    end_t = timeit.default_timer()
                    timings.append(end_t - beg_t)
                    beg_t = timeit.default_timer()
                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, ' \
                          'd_fake:{:.4f}, '.format(global_step, bidx,
                                                   len(dloader), epoch,
                                                   d_real_loss_v,
                                                   d_fake_loss_v)
                    if not self.pesq_objective:
                        log += 'd_ma_fake:{:.4f}, '.format(d_ma_fake_loss_v)
                    
                    log += 'g_adv:{:.4f}, g_l1:{:.4f} ' \
                           'l1_w: {:.3f}, btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(g_adv_loss_v,
                                     g_l1_loss_v, l1_weight, timings[-1],
                                     np.mean(timings))
                    print(log)
                    self.writer.add_scalar('D_real', d_real_loss_v,
                                           global_step)
                    self.writer.add_scalar('D_fake', d_fake_loss_v,
                                           global_step)
                    if not self.pesq_objective:
                        self.writer.add_scalar('D_ma_fake', d_ma_fake_loss_v,
                                               global_step)
                    self.writer.add_scalar('G_adv', g_adv_loss_v,
                                           global_step)
                    self.writer.add_scalar('G_l1', g_l1_loss_v,
                                           global_step)
                    self.writer.add_histogram('Gz', Genh.cpu().data.numpy(),
                                              global_step, bins='sturges')
                    self.writer.add_histogram('clean', clean.cpu().data.numpy(),
                                              global_step, bins='sturges')
                    self.writer.add_histogram('noisy', noisy.cpu().data.numpy(),
                                              global_step, bins='sturges')
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
                                                       global_step)
                                total_GW_norm += W_norm
                        self.writer.add_scalar('{}_Wnorm'.format(total_name),
                                               total_GW_norm,
                                               global_step)
                    model_weights_norm(self.G, 'Gtotal')
                    model_weights_norm(self.D, 'Dtotal')
                    canvas_w = self.G(noisy_samples, z=z_sample)
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
                                                   '{}.wav'.format(global_step,
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
                    # save model
                    self.save(self.save_path, global_step)
                global_step += 1

            if va_dloader is not None:
                #pesqs, mpesq = self.evaluate(opts, va_dloader, log_freq)
                pesqs, npesqs, \
                mpesq, mnpesq, \
                ssnrs, nssnrs, \
                mssnr, mnssnr = self.evaluate(opts, va_dloader, 
                                              log_freq, do_noisy=True)
                print('mean noisyPESQ: ', mnpesq)
                print('mean GenhPESQ: ', mpesq)
                self.writer.add_scalar('noisyPESQ', mnpesq, epoch)
                self.writer.add_scalar('noisySSNR', mnssnr, epoch)
                self.writer.add_scalar('GenhPESQ', mpesq, epoch)
                self.writer.add_scalar('GenhSSNR', mssnr, epoch)
                #self.writer.add_histogram('noisyPESQ', npesqs,
                #                          epoch, bins='sturges')
                #self.writer.add_histogram('GenhPESQ', pesqs,
                #                          epoch, bins='sturges')


    def evaluate(self, opts, dloader, log_freq, do_noisy=False,
                 max_samples=100):
        """ Objective evaluation with PESQ and SSNR """
        self.G.eval()
        self.D.eval()
        beg_t = timeit.default_timer()
        pesqs = []
        ssnrs = []
        if do_noisy:
            npesqs = []
            nssnrs = []
        total_s = 0
        timings = []
        # going over dataset ONCE
        for bidx, batch in enumerate(dloader, start=1):
            sample = batch
            if len(sample) == 2:
                clean, noisy = batch
            elif len(sample) == 3:
                clean, noisy, pesq = batch
            else:
                clean, noisy, pesq, ssnr = batch
            clean = Variable(clean, volatile=True)
            noisy = Variable(noisy.unsqueeze(1), volatile=True)
            if self.do_cuda:
                clean = clean.cuda()
                noisy = noisy.cuda()
            Genh = self.G(noisy).squeeze(1)
            clean_npy = clean.cpu().data.numpy()
            if do_noisy:
                noisy_npy = noisy.cpu().data.numpy()
            Genh_npy = Genh.cpu().data.numpy()
            for sidx in range(Genh.size(0)):
                clean_utt = denormalize_wave_minmax(clean_npy[sidx]).astype(np.int16)
                clean_utt = clean_utt.reshape(-1)
                clean_utt = de_emphasize(clean_utt, self.preemph)
                Genh_utt = denormalize_wave_minmax(Genh_npy[sidx]).astype(np.int16)
                Genh_utt = Genh_utt.reshape(-1)
                Genh_utt = de_emphasize(Genh_utt, self.preemph)
                # compute PESQ per file
                pesq = PESQ(clean_utt, Genh_utt)
                if 'error' in pesq:
                    print('Skipping error')
                    continue
                pesq = float(pesq)
                pesqs.append(pesq)
                snr_mean, segsnr_mean = SSNR(clean_utt, Genh_utt)
                segsnr_mean = float(segsnr_mean)
                ssnrs.append(segsnr_mean)
                print('Genh sample {} > PESQ: {:.3f}, SSNR: {:.3f} dB'
                      ''.format(total_s, pesq, segsnr_mean))
                if do_noisy:
                    # noisy PESQ too
                    noisy_utt = denormalize_wave_minmax(noisy_npy[sidx]).astype(np.int16)
                    noisy_utt = noisy_utt.reshape(-1)
                    noisy_utt = de_emphasize(noisy_utt, self.preemph)
                    npesq = PESQ(clean_utt, noisy_utt)
                    npesq = float(npesq)
                    npesqs.append(npesq)
                    nsnr_mean, nsegsnr_mean = SSNR(clean_utt, noisy_utt)
                    nsegsnr_mean = float(nsegsnr_mean)
                    nssnrs.append(nsegsnr_mean)
                    print('Noisy sample {} > PESQ: {:.3f}, SSNR: {:.3f} dB'
                          ''.format(total_s, npesq, nsegsnr_mean))
                # Segmental SNR
                total_s += 1
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                print('Mean pesq computation time: {}'
                      's'.format(np.mean(timings)))
                beg_t = timeit.default_timer()
                #print('{} PESQ: {}'.format(sidx, pesq))
                #wavfile.write('{}_clean_test.wav'.format(sidx), 16000,
                #              clean_utt)
                #wavfile.write('{}_enh_test.wav'.format(sidx), 16000,
                #              Genh_utt)
            #if bidx % log_freq == 0 or bidx >= len(dloader):
            #    print('EVAL Batch {}/{} mPESQ: {:.4f}'
            #          ''.format(bidx,
            #                    len(dloader),
            #                    np.mean(pesqs)))
            if total_s >= max_samples:
                break
        return np.array(pesqs), np.array(npesqs), np.mean(pesqs), \
               np.mean(npesqs), np.array(ssnrs), \
               np.array(nssnrs), np.mean(ssnrs), np.mean(nssnrs)


class SilentSEGAN(Model):

    def __init__(self, opts, name='SilentSEGAN'):
        super(SilentSEGAN, self).__init__(name)
        self.opts = opts
        # max_pad operates on G input to translate signal here and there
        self.max_pad = opts.max_pad
        self.preemph = opts.preemph
        self.save_path = opts.save_path
        self.do_cuda = opts.cuda
        self.g_dropout = opts.g_dropout
        self.z_dim = opts.z_dim
        self.g_enc_fmaps = opts.g_enc_fmaps
        self.g_bias = opts.g_bias
        self.max_ma = opts.max_ma
        self.stereo_D = opts.stereo_D
        self.num_spks = opts.num_spks
        self.BID = opts.BID
        self.d_optim = opts.d_optim
        self.g_optim = opts.g_optim
        self.d_noise_std = opts.d_noise_std
        self.d_noise_epoch = opts.d_noise_epoch
        self.d_real_weight = opts.d_real_weight
        self.g_weight = opts.g_weight
        self.d_fake_weight = opts.d_fake_weight
        self.noise_dec_step = opts.noise_dec_step
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
        elif opts.g_act == 'glu':
            self.g_enc_act = 'glu'
            self.g_dec_act = None
        else:
            raise TypeError('Unrecognized G activation: ', opts.g_act)

        # Build G and D
        self.G = Generator1D(1, self.g_enc_fmaps, opts.kwidth,
                             self.g_enc_act,
                             lnorm=opts.g_bnorm, dropout=opts.g_dropout, 
                             pooling=2, z_dim=opts.z_dim,
                             z_all=opts.z_all,
                             cuda=opts.cuda,
                             skip=opts.skip,
                             skip_blacklist=opts.skip_blacklist,
                             dec_activations=self.g_dec_act,
                             bias=opts.g_bias,
                             aal=opts.g_aal,
                             aal_out=opts.g_aal_out,
                             wd=opts.wd,
                             skip_init=opts.skip_init,
                             skip_dropout=opts.skip_dropout,
                             no_tanh=opts.no_tanh,
                             rnn_core=opts.g_rnn_core,
                             linterp=opts.linterp,
                             mlpconv=opts.g_mlpconv,
                             dec_kwidth=opts.dec_kwidth,
                             subtract_mean=opts.g_subtract_mean,
                             no_z=opts.no_z)
                             
        if not opts.no_winit:
            self.G.apply(weights_init)
        print('Generator: ', self.G)

        self.d_enc_fmaps = opts.d_enc_fmaps
        if self.BID:
            self.D = BiDiscriminator(self.d_enc_fmaps, opts.kwidth,
                                     nn.LeakyReLU(0.2, inplace=True),
                                     bnorm=opts.d_bnorm,
                                     pooling=2, SND=opts.SND,
                                     dropout=opts.d_dropout)
        else:
            if self.stereo_D:
                D_in = 2
            else:
                D_in = 1
            if opts.DG_tied:
                Genc = self.G.gen_enc
            else:
                Genc = None
            self.D = Discriminator(D_in, self.d_enc_fmaps, opts.kwidth,
                                   nn.LeakyReLU(0.2, inplace=True),
                                   bnorm=opts.d_bnorm,
                                   pooling=2, SND=opts.SND,
                                   pool_type=opts.D_pool_type,
                                   dropout=opts.d_dropout,
                                   Genc=Genc, 
                                   pool_size=opts.D_pool_size,
                                   num_spks=self.num_spks)
        if not opts.no_winit:
            self.D.apply(weights_init)
        print('Discriminator: ', self.D)
        if self.do_cuda:
            self.D.cuda()
            self.G.cuda()
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))


    def generate(self, inwav, ret_hid=False):
        return self.G(inwav, ret_hid=ret_hid)

    def discriminate(self, inwav):
        return self.D(inwav)

    def train(self, opts, dloader, criterion, 
              log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        if self.g_optim == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
        else:
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr,
                               betas=(0.7, 0.9))
        if opts.g_step_lr is not None:
            # apply lr annealing
            g_sched = lr_scheduler.StepLR(Gopt, opts.g_step_lr,
                                          opts.g_lr_gamma)
        else:
            g_sched = None

        if self.d_optim == 'rmsprop':
            Dopt = optim.RMSprop(self.D.parameters(), lr=opts.d_lr)
        else:
            Dopt = optim.Adam(self.D.parameters(), lr=opts.d_lr,
                               betas=(0.7, 0.9))
        if opts.d_step_lr is not None:
            # apply lr annealing
            d_sched = lr_scheduler.StepLR(Dopt, opts.d_step_lr,
                                          opts.d_lr_gamma)
        else:
            d_sched = None


        num_batches = len(dloader) 

        self.G.train()
        self.D.train()

        global_step = 1
        timings = []
        noisy_samples = None
        clean_samples = None
        z_sample = None
        d_noise_std = self.d_noise_std
        d_noise_epoch = self.d_noise_epoch
        noise_dec_step = self.noise_dec_step
        D_x = None
        D_G_z1 = None
        D_G_z2 = None
        spkid = None
        l1_weight = opts.l1_weight
        for epoch in range(1, opts.epoch + 1):
            # possibly update learning rates
            if g_sched is not None:
                G_lr = Gopt.param_groups[0]['lr']
                # WARNING: This does not ensure we will have g_min_lr
                # BUT we will have a maximum update to downgrade lr prior
                # to certain minima
                if G_lr > opts.g_min_lr:
                    g_sched.step()
            if d_sched is not None:
                D_lr = Dopt.param_groups[0]['lr']
                # WARNING: This does not ensure we will have g_min_lr
                # BUT we will have a maximum update to downgrade lr prior
                # to certain minima
                if D_lr > opts.d_min_lr:
                    d_sched.step()
            if epoch >= opts.l1_dec_epoch:
                if l1_weight > 0:
                    l1_weight -= opts.l1_dec_step
                    # ensure it is 0 if it goes < 0
                    l1_weight = max(0, l1_weight)
            beg_t = timeit.default_timer()
            for bidx, batch in enumerate(dloader, start=1):
                sample = batch
                if len(sample) == 2:
                    clean, noisy = batch
                elif len(sample) == 3:
                    if self.num_spks is None:
                        raise ValueError('Need num_spks active in SilentSEGAN'\
                                         ' when delivering spkid in dataloader')
                    clean, noisy, spkid = batch
                    spkid = Variable(spkid).view(-1)
                    if self.do_cuda:
                        spkid = spkid.cuda()
                clean = Variable(clean.unsqueeze(1))
                noisy = Variable(noisy.unsqueeze(1))
                d_weight = 1.
                lab = Variable((1 - smooth) * torch.ones(clean.size(0), 1))
                if self.do_cuda:
                    clean = clean.cuda()
                    noisy = noisy.cuda()
                    lab = lab.cuda()
                # shift slightly noisy as input to G
                if self.max_pad > 0:
                    pad_size = np.random.randint(0, self.max_pad)
                    left = np.random.rand(1)[0]
                    if left > 0.5:
                        # pad left
                        pad = (pad_size, 0)
                    else:
                        # pad right
                        pad = (0, pad_size)
                    pnoisy = F.pad(noisy, pad, mode='reflect')
                    pclean = F.pad(clean, pad, mode='reflect')
                    if left:
                        noisy = pnoisy[:, :, :noisy.size(-1)].contiguous()
                        clean = pclean[:, :, :clean.size(-1)].contiguous()
                    else:
                        noisy = pnoisy[:, :, -noisy.size(-1):].contiguous()
                        clean = pclean[:, :, -clean.size(-1):].contiguous()
                else:
                    pad_size = 0
                #print('clean size: ', clean.size())
                #print('noisy size: ', clean.size())
                if noisy_samples is None:
                    noisy_samples = noisy[:20, :, :]
                    clean_samples = clean[:20, :, :]
                N_noise = Variable(torch.randn(clean.size()) * d_noise_std)
                if self.do_cuda:
                    N_noise = N_noise.cuda()
                clean = clean + N_noise
                # (1) D real update
                Dopt.zero_grad()
                if self.stereo_D or self.BID:
                    D_in = torch.cat((clean, noisy), dim=1)
                else:
                    D_in = clean
                if self.BID:
                    lab = Variable(lab.data.fill_(1))
                    # do cosine similarity loss
                    _, d_real_1, d_real_2, d_real_iact = self.D(D_in)
                    d_real_loss = self.d_real_weight * \
                                  F.cosine_embedding_loss(d_real_1, d_real_2, lab)
                else:
                    d_real, d_real_iact= self.D(D_in)
                    if spkid is not None:
                        d_spk = d_real[:, 1:].contiguous()
                        d_real = d_real[:, :1].contiguous()
                    d_real_loss = criterion(d_real, lab)
                    D_x = F.sigmoid(d_real).data.mean()
                    if spkid is not None:
                        d_spkid_loss = F.cross_entropy(d_spk, spkid)
                        d_real_loss = d_real_loss + d_spkid_loss
                    d_real_loss = self.d_real_weight * d_real_loss
                #print('d_real size: ', d_real.size())
                d_real_loss.backward()
                
                # (2) D fake update
                Genh = self.G(noisy)
                N_noise = Variable(torch.randn(Genh.size()) * d_noise_std)
                if self.do_cuda:
                    N_noise = N_noise.cuda()
                Genh = Genh + N_noise
                if self.stereo_D or self.BID:
                    D_fake_in = torch.cat((Genh.detach(), noisy), dim=1)
                else:
                    D_fake_in = Genh.detach()
                if self.BID:
                    lab = Variable(lab.data.fill_(-1))
                    # do cosine similarity loss
                    _, d_fake_1, d_fake_2, d_fake_iact = self.D(D_fake_in)
                    d_fake_loss = self.d_fake_weight * \
                                  F.cosine_embedding_loss(d_fake_1, d_fake_2,
                                                          lab)
                else:
                    lab = Variable(lab.data.fill_(0))
                    d_fake, d_fake_iact = self.D(D_fake_in)
                    if spkid is not None:
                        d_fake = d_fake[:, :1].contiguous()
                    #print('d_fake size: ', d_fake.size())
                    d_fake_loss = self.d_fake_weight * criterion(d_fake, lab)
                    D_G_z1 = F.sigmoid(d_fake).data.mean()
                d_fake_loss.backward()
                Dopt.step()

                d_loss = d_fake_loss  + d_real_loss

                # (3) G real update
                Gopt.zero_grad()
                g_weight = self.g_weight
                lab = Variable(lab.data.fill_(1 - smooth))
                if self.stereo_D or self.BID:
                    d_fake__in = torch.cat((Genh, noisy), dim=1)
                else:
                    d_fake__in = Genh
                if self.BID:
                    lab = Variable(lab.data.fill_(1))
                    # do cosine similarity loss
                    _, d_fake__1, d_fake__2, d_fake__iact = self.D(d_fake__in)
                    g_adv_loss = self.g_weight * \
                                 F.cosine_embedding_loss(d_fake__1, d_fake__2,
                                                         lab)
                else:
                    d_fake_, d_fake__iact = self.D(d_fake__in)
                    if spkid is not None:
                        d_spk_ = d_fake_[:, 1:].contiguous()
                        d_fake_ = d_fake_[:, :1].contiguous()
                    #print('d_fake_ size: ', d_fake_.size())
                    #print('d_fake_ max: ', d_fake_.max())
                    #print('d_fake_ min: ', d_fake_.min())
                    g_adv_loss = criterion(d_fake_, lab)
                    if spkid is not None:
                        g_spkid_loss = F.cross_entropy(d_spk_, spkid)
                        g_adv_loss = g_adv_loss + g_spkid_loss
                    g_adv_loss = g_weight * g_adv_loss
                    D_G_z2 = F.sigmoid(d_fake_).data.mean()
                g_l1_loss = l1_weight * F.l1_loss(Genh, clean)
                g_loss = g_adv_loss + g_l1_loss
                g_loss.backward()
                Gopt.step()
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if z_sample is None and not self.G.no_z:
                    # capture sample now that we know shape after first
                    # inference
                    if isinstance(self.G.z, tuple):
                        z_sample = (self.G.z[0][:, :20 ,:].contiguous(),
                                    self.G.z[1][:, :20 ,:].contiguous())
                        print('Getting sampling z with size: ', z_sample[0].size())
                    else:
                        z_sample = self.G.z[:20, : ,:].contiguous()
                        print('Getting sampling z with size: ', z_sample.size())
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    if epoch >= d_noise_epoch and d_noise_std > 0:
                        d_noise_std -= noise_dec_step
                        d_noise_std = max(0, d_noise_std)
                    d_real_loss_v = np.asscalar(d_real_loss.cpu().data.numpy())
                    d_fake_loss_v = np.asscalar(d_fake_loss.cpu().data.numpy())
                    g_adv_loss_v = np.asscalar(g_adv_loss.cpu().data.numpy())
                    g_l1_loss_v = np.asscalar(g_l1_loss.cpu().data.numpy())
                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, ' \
                          'd_fake:{:.4f}, '.format(global_step, bidx,
                                                   len(dloader), epoch,
                                                   d_real_loss_v,
                                                   d_fake_loss_v)
                    if self.num_spks is not None:
                        log += 'd_real_spk:{:.4f}, g_spk:{:.4f}, ' \
                               ''.format(d_spkid_loss.cpu().data[0],
                                         g_spkid_loss.cpu().data[0])
                    log += 'g_adv:{:.4f}, g_l1:{:.4f}, l1_w:{:.2f}, '\
                           'd_noise_std:{:.3f}' \
                           ' btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(g_adv_loss_v,
                                     g_l1_loss_v,
                                     l1_weight,
                                     d_noise_std,
                                     timings[-1],
                                     np.mean(timings))
                    G_lr = Gopt.param_groups[0]['lr']
                    D_lr = Dopt.param_groups[0]['lr']
                    log += ', G_curr_lr: ' \
                           '{:.6f}'.format(G_lr)
                    log += ', D_curr_lr: ' \
                           '{:.6f}'.format(D_lr)
                    print(log)
                    if self.num_spks is not None:
                        self.writer.add_scalar('D_spk_loss',
                                               d_spkid_loss.cpu().data[0], global_step)
                        self.writer.add_scalar('G_spk_loss',
                                               g_spkid_loss.cpu().data[0], global_step)
                    self.writer.add_scalar('G_lr', G_lr, global_step)
                    self.writer.add_scalar('D_lr', D_lr, global_step)
                    if D_x is not None:
                        self.writer.add_scalar('D_x', D_x,
                                               global_step)
                        self.writer.add_scalar('D_G_z1', D_G_z1,
                                               global_step)
                        self.writer.add_scalar('D_G_z2', D_G_z2,
                                               global_step)
                        self.writer.add_scalar('D(G(z))', D_G_z1 / (D_G_z2 + 1e-10),
                                               global_step)
                        self.writer.add_histogram('D_real_logits', d_real.cpu().data,
                                                  global_step, bins='sturges')
                        self.writer.add_histogram('D_fake_logits', d_fake.cpu().data,
                                                  global_step, bins='sturges')
                        self.writer.add_histogram('D_fake__logits', d_fake_.cpu().data,
                                                  global_step, bins='sturges')
                    self.writer.add_scalar('d_noise_std', d_noise_std,
                                           global_step)
                    self.writer.add_scalar('D_real', d_real_loss_v,
                                           global_step)
                    self.writer.add_scalar('D_fake', d_fake_loss_v,
                                           global_step)
                    self.writer.add_scalar('G_adv', g_adv_loss_v,
                                           global_step)
                    self.writer.add_scalar('G_l1', g_l1_loss_v,
                                           global_step)
                    self.writer.add_scalar('l1_weight', l1_weight, global_step)
                    self.writer.add_histogram('Gz', Genh.cpu().data.numpy(),
                                              global_step, bins='sturges')
                    self.writer.add_histogram('clean', clean.cpu().data.numpy(),
                                              global_step, bins='sturges')
                    self.writer.add_histogram('noisy', noisy.cpu().data.numpy(),
                                              global_step, bins='sturges')
                    self.writer.add_scalar('G_noisy_pad_size', 
                                           pad_size,
                                           global_step)
                    # histogram of intermediate activations from d_real and
                    # d_fake
                    for k, v in d_real_iact.items():
                        self.writer.add_histogram(k, v.cpu().data, global_step,
                                                  bins='sturges')
                    for k, v in d_fake_iact.items():
                        self.writer.add_histogram(k, v.cpu().data, global_step,
                                                  bins='sturges')
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
                                                       global_step)
                                total_GW_norm += W_norm
                        self.writer.add_scalar('{}_Wnorm'.format(total_name),
                                               total_GW_norm,
                                               global_step)
                    model_weights_norm(self.G, 'Gtotal')
                    model_weights_norm(self.D, 'Dtotal')
                    # Plot G skip connection parameters
                    for skip_idx, skip_conn in self.G.skips.items():
                        alpha = skip_conn['alpha'].cpu().data
                        self.writer.add_scalar('skip_{}_a-norm'.format(skip_idx),
                                               torch.norm(alpha),
                                               global_step)
                        self.writer.add_histogram('skip_{}_a'.format(skip_idx),
                                                  alpha,
                                                  global_step, bins='sturges')
                    canvas_w = self.G(noisy_samples, z=z_sample)
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
                                                   '{}.wav'.format(global_step,
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
                    # save model
                    self.save(self.save_path, global_step)
                global_step += 1

            if va_dloader is not None:
                #pesqs, mpesq = self.evaluate(opts, va_dloader, log_freq)
                D_xs, D_G_zs = self.evaluate(opts, va_dloader, criterion, epoch)


    def evaluate(self, opts, dloader, criterion, epoch, max_samples=100):
        """ Evaluate D_x, G_z1 and G_z2 with validation data """
        self.G.eval()
        self.D.eval()
        beg_t = timeit.default_timer()
        total_s = 0
        timings = []
        D_xs = []
        D_G_zs = []
        G_fake_losses = []
        D_real_losses = []
        G_spk_losses = []
        D_spk_losses = []
        # going over dataset ONCE
        for bidx, batch in enumerate(dloader, start=1):
            sample = batch
            if len(sample) == 2:
                clean, noisy = batch
            elif len(sample) == 3:
                if self.num_spks is None:
                    raise ValueError('Need num_spks active in SilentSEGAN'\
                                     ' when delivering spkid in dataloader')
                clean, noisy, spkid = batch
                spkid = Variable(spkid).view(-1)
                if self.do_cuda:
                    spkid = spkid.cuda()
            clean = Variable(clean.unsqueeze(1), volatile=True)
            noisy = Variable(noisy.unsqueeze(1), volatile=True)
            if self.do_cuda:
                clean = clean.cuda()
                noisy = noisy.cuda()
            # prepare lab to compute eval loss only with true target
            lab = Variable(torch.ones(clean.size(0), 1))
            if self.do_cuda:
                lab = lab.cuda()
            # forward through G
            Genh = self.G(noisy)
            # forward real and fake through D
            if self.stereo_D:
                D_real_in = torch.cat((clean, noisy), dim=1)
            else:
                D_real_in = clean
            d_real, d_real_iact= self.D(D_real_in)
            if spkid is not None:
                d_real_spk = d_real[:, 1:].contiguous()
                d_real = d_real[:, :1].contiguous()
                d_real_spkid_loss = F.cross_entropy(d_real_spk, spkid)
                D_spk_losses.append(d_real_spkid_loss.data[0])
            d_real_loss = criterion(d_real, lab)
            D_real_losses.append(d_real_loss.data[0])
            if self.stereo_D:
                D_fake_in = torch.cat((Genh.detach(), noisy), dim=1)
            else:
                D_fake_in = Genh.detach()
            d_fake, d_fake_iact= self.D(D_fake_in)
            if spkid is not None:
                d_fake_spk = d_fake[:, 1:].contiguous()
                d_fake = d_fake[:, :1].contiguous()
                d_fake_spkid_loss = F.cross_entropy(d_fake_spk, spkid)
                G_spk_losses.append(d_fake_spkid_loss.data[0])
            d_fake_loss = criterion(d_fake, lab)
            G_fake_losses.append(d_fake_loss.data[0])
            D_x = F.sigmoid(d_real).cpu().data
            D_G_z = F.sigmoid(d_fake).cpu().data
            D_xs.append(D_x)
            D_G_zs.append(D_G_z)
            
            if total_s >= max_samples:
                break
        D_real_losses = torch.FloatTensor(D_real_losses)
        G_fake_losses = torch.FloatTensor(G_fake_losses)
        if len(G_spk_losses) > 0:
            G_spk_losses = torch.FloatTensor(G_spk_losses)
            D_spk_losses = torch.FloatTensor(D_spk_losses)
        D_xs = torch.cat(D_xs, dim=0)
        D_G_zs = torch.cat(D_G_zs, dim=0)
        self.writer.add_histogram('Eval-D_x', D_xs,
                                  epoch, bins='sturges')
        self.writer.add_histogram('Eval-D_G_z', D_G_zs, 
                                  epoch, bins='sturges')
        self.writer.add_scalar('meanEval-D_G_z', D_G_zs.mean(), 
                               epoch)
        self.writer.add_scalar('meanEval-D_x', D_xs.mean(), 
                               epoch)
        self.writer.add_scalar('meanEval-D_real_loss',
                               D_real_losses.mean(), epoch)
        # renamed to G_real because it is when G wants to fake D,
        # namings are correct although swapping
        self.writer.add_scalar('meanEval-G_real_loss',
                               G_fake_losses.mean(), epoch)
        self.writer.add_histogram('Eval-D_real_loss',
                                   D_real_losses, epoch,
                                   bins='sturges')
        self.writer.add_histogram('Eval-G_real_loss',
                                   G_fake_losses, epoch,
                                   bins='sturges')
        if self.num_spks is not None:
            self.writer.add_scalar('meanEval-D_spk_loss',
                                   D_spk_losses.mean(), epoch)
            self.writer.add_scalar('meanEval-G_spk_loss',
                                   G_spk_losses.mean(), epoch)
            self.writer.add_histogram('Eval-G_spk_loss',
                                       G_spk_losses, epoch,
                                       bins='sturges')
            self.writer.add_histogram('Eval-D_spk_loss',
                                       D_spk_losses, epoch,
                                       bins='sturges')
        return D_xs, D_G_zs

class CycleSEGAN(Model):

    def __init__(self, opts, name='SilentSEGAN'):
        super(SilentSEGAN, self).__init__(name)
        self.opts = opts
        # max_pad operates on G input to translate signal here and there
        self.max_pad = opts.max_pad
        self.preemph = opts.preemph
        self.save_path = opts.save_path
        self.do_cuda = opts.cuda
        self.z_dim = opts.z_dim
        self.g_enc_fmaps = opts.g_enc_fmaps
        self.max_ma = opts.max_ma
        self.BID = opts.BID
        self.d_optim = opts.d_optim
        self.g_optim = opts.g_optim
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
        elif opts.g_act == 'glu':
            self.g_enc_act = 'glu'
            self.g_dec_act = None
        else:
            raise TypeError('Unrecognized G activation: ', opts.g_act)

        # Build GA and GB
        self.G_A = Generator1D(1, self.g_enc_fmaps, opts.kwidth,
                               self.g_enc_act,
                               lnorm=opts.g_bnorm, dropout=0,
                               pooling=2, z_dim=opts.z_dim,
                               z_all=opts.z_all,
                               cuda=opts.cuda,
                               skip=opts.skip,
                               skip_blacklist=opts.skip_blacklist,
                               dec_activations=self.g_dec_act,
                               bias=opts.g_bias,
                               aal=False,
                               aal_out=False,
                               wd=0,
                               skip_init=opts.skip_init,
                               skip_dropout=opts.skip_dropout,
                               no_tanh=opts.no_tanh,
                               rnn_core=opts.g_rnn_core,
                               linterp=opts.linterp,
                               mlpconv=opts.g_mlpconv,
                               dec_kwidth=opts.dec_kwidth,
                               subtract_mean=False,
                               no_z=opts.no_z)
        self.G_B = Generator1D(1, self.g_enc_fmaps, opts.kwidth,
                               self.g_enc_act,
                               lnorm=opts.g_bnorm, dropout=0,
                               pooling=2, z_dim=opts.z_dim,
                               z_all=opts.z_all,
                               cuda=opts.cuda,
                               skip=opts.skip,
                               skip_blacklist=opts.skip_blacklist,
                               dec_activations=self.g_dec_act,
                               bias=opts.g_bias,
                               aal=False,
                               aal_out=False,
                               wd=0,
                               skip_init=opts.skip_init,
                               skip_dropout=opts.skip_dropout,
                               no_tanh=opts.no_tanh,
                               rnn_core=opts.g_rnn_core,
                               linterp=opts.linterp,
                               mlpconv=opts.g_mlpconv,
                               dec_kwidth=opts.dec_kwidth,
                               subtract_mean=False,
                               no_z=opts.no_z)
                             
        if not opts.no_winit:
            self.G.apply(weights_init)
        print('Generator: ', self.G)

        self.d_enc_fmaps = opts.d_enc_fmaps
        self.D_A = Discriminator(1, self.d_enc_fmaps, opts.kwidth,
                                 nn.LeakyReLU(0.2, inplace=True),
                                 bnorm=opts.d_bnorm,
                                 pooling=2, SND=opts.SND,
                                 pool_type=opts.D_pool_type,
                                 dropout=0,
                                 pool_size=opts.D_pool_size)
        if not opts.no_winit:
            self.D.apply(weights_init)
        print('Discriminator: ', self.D)
        if self.do_cuda:
            self.D.cuda()
            self.G.cuda()
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))

    def train(self, opts, dloader, criterion, 
              log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        if self.g_optim == 'rmsprop':
            Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
        else:
            Gopt = optim.Adam(self.G.parameters(), lr=opts.g_lr,
                               betas=(0.7, 0.9))
        if self.d_optim == 'rmsprop':
            Dopt = optim.RMSprop(self.D.parameters(), lr=opts.d_lr)
        else:
            Dopt = optim.Adam(self.D.parameters(), lr=opts.d_lr,
                               betas=(0.7, 0.9))

        num_batches = len(dloader) 

        self.G.train()
        self.D.train()

        global_step = 1
        timings = []
        noisy_samples = None
        clean_samples = None
        z_sample = None
        d_noise_std = self.d_noise_std
        d_noise_epoch = self.d_noise_epoch
        noise_dec_step = self.noise_dec_step
        D_x = None
        D_G_z1 = None
        D_G_z2 = None
        for epoch in range(1, opts.epoch + 1):
            beg_t = timeit.default_timer()
            for bidx, batch in enumerate(dloader, start=1):
                sample = batch
                if len(sample) == 2:
                    clean, noisy = batch
                elif len(sample) == 3:
                    clean, noisy, pesq = batch
                else:
                    clean, noisy, pesq, ssnr = batch
                clean = Variable(clean.unsqueeze(1))
                noisy = Variable(noisy.unsqueeze(1))
                d_weight = 1.
                lab = Variable((1 - smooth) * torch.ones(clean.size(0), 1))
                if self.do_cuda:
                    clean = clean.cuda()
                    noisy = noisy.cuda()
                    lab = lab.cuda()
                # shift slightly noisy as input to G
                if self.max_pad > 0:
                    pad_size = np.random.randint(0, self.max_pad)
                    left = np.random.rand(1)[0]
                    if left > 0.5:
                        # pad left
                        pad = (pad_size, 0)
                    else:
                        # pad right
                        pad = (0, pad_size)
                    pnoisy = F.pad(noisy, pad, mode='reflect')
                    pclean = F.pad(clean, pad, mode='reflect')
                    if left:
                        noisy = pnoisy[:, :, :noisy.size(-1)].contiguous()
                        clean = pclean[:, :, :clean.size(-1)].contiguous()
                    else:
                        noisy = pnoisy[:, :, -noisy.size(-1):].contiguous()
                        clean = pclean[:, :, -clean.size(-1):].contiguous()
                else:
                    pad_size = 0
                #print('clean size: ', clean.size())
                #print('noisy size: ', clean.size())
                if noisy_samples is None:
                    noisy_samples = noisy[:20, :, :]
                    clean_samples = clean[:20, :, :]
                N_noise = Variable(torch.randn(clean.size()) * d_noise_std)
                if self.do_cuda:
                    N_noise = N_noise.cuda()
                clean = clean + N_noise
                # (1) D real update
                Dopt.zero_grad()
                if self.stereo_D or self.BID:
                    D_in = torch.cat((clean, noisy), dim=1)
                else:
                    D_in = clean
                if self.BID:
                    lab = Variable(lab.data.fill_(1))
                    # do cosine similarity loss
                    _, d_real_1, d_real_2, d_real_iact = self.D(D_in)
                    d_real_loss = self.d_real_weight * \
                                  F.cosine_embedding_loss(d_real_1, d_real_2, lab)
                else:
                    d_real, d_real_iact= self.D(D_in)
                    d_real_loss = self.d_real_weight * criterion(d_real, lab)
                    D_x = d_real.data.mean()
                #print('d_real size: ', d_real.size())
                d_real_loss.backward()
                
                # (2) D fake update
                Genh = self.G(noisy)
                N_noise = Variable(torch.randn(Genh.size()) * d_noise_std)
                if self.do_cuda:
                    N_noise = N_noise.cuda()
                Genh = Genh + N_noise
                if self.stereo_D or self.BID:
                    D_fake_in = torch.cat((Genh.detach(), noisy), dim=1)
                else:
                    D_fake_in = Genh.detach()
                if self.BID:
                    lab = Variable(lab.data.fill_(-1))
                    # do cosine similarity loss
                    _, d_fake_1, d_fake_2, d_fake_iact = self.D(D_fake_in)
                    d_fake_loss = self.d_fake_weight * \
                                  F.cosine_embedding_loss(d_fake_1, d_fake_2,
                                                          lab)
                else:
                    lab = Variable(lab.data.fill_(0))
                    d_fake, d_fake_iact = self.D(D_fake_in)
                    #print('d_fake size: ', d_fake.size())
                    d_fake_loss = self.d_fake_weight * criterion(d_fake, lab)
                    D_G_z1 = d_fake.data.mean()
                d_fake_loss.backward()
                Dopt.step()

                d_loss = d_fake_loss  + d_real_loss

                # (3) G real update
                Gopt.zero_grad()
                g_weight = self.g_weight
                lab = Variable(lab.data.fill_(1 - smooth))
                if self.stereo_D or self.BID:
                    d_fake__in = torch.cat((Genh, noisy), dim=1)
                else:
                    d_fake__in = Genh
                if self.BID:
                    lab = Variable(lab.data.fill_(1))
                    # do cosine similarity loss
                    _, d_fake__1, d_fake__2, d_fake__iact = self.D(d_fake__in)
                    g_adv_loss = self.g_weight * \
                                 F.cosine_embedding_loss(d_fake__1, d_fake__2,
                                                         lab)
                else:
                    d_fake_, d_fake__iact = self.D(d_fake__in)
                    #print('d_fake_ size: ', d_fake_.size())
                    #print('d_fake_ max: ', d_fake_.max())
                    #print('d_fake_ min: ', d_fake_.min())
                    g_adv_loss = g_weight * criterion(d_fake_, lab)
                    D_G_z2 = d_fake_.data.mean()
                g_loss = g_adv_loss
                g_loss.backward()
                Gopt.step()
                if z_sample is None and not self.G.no_z:
                    # capture sample now that we know shape after first
                    # inference
                    if isinstance(self.G.z, tuple):
                        z_sample = (self.G.z[0][:, :20 ,:].contiguous(),
                                    self.G.z[1][:, :20 ,:].contiguous())
                        print('Getting sampling z with size: ', z_sample[0].size())
                    else:
                        z_sample = self.G.z[:20, : ,:].contiguous()
                        print('Getting sampling z with size: ', z_sample.size())
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    if epoch >= d_noise_epoch and d_noise_std > 0:
                        d_noise_std -= noise_dec_step
                        d_noise_std = max(0, d_noise_std)
                    d_real_loss_v = np.asscalar(d_real_loss.cpu().data.numpy())
                    d_fake_loss_v = np.asscalar(d_fake_loss.cpu().data.numpy())
                    g_adv_loss_v = np.asscalar(g_adv_loss.cpu().data.numpy())
                    end_t = timeit.default_timer()
                    timings.append(end_t - beg_t)
                    beg_t = timeit.default_timer()
                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, ' \
                          'd_fake:{:.4f}, '.format(global_step, bidx,
                                                   len(dloader), epoch,
                                                   d_real_loss_v,
                                                   d_fake_loss_v)
                    log += 'g_adv:{:.4f}, d_noise_std:{:.3f}' \
                           'btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(g_adv_loss_v,
                                     d_noise_std,
                                     timings[-1],
                                     np.mean(timings))
                    print(log)
                    if D_x is not None:
                        self.writer.add_scalar('D_x', D_x,
                                               global_step)
                        self.writer.add_scalar('D_G_z1', D_G_z1,
                                               global_step)
                        self.writer.add_scalar('D_G_z2', D_G_z2,
                                               global_step)
                        self.writer.add_scalar('D(G(z))', D_G_z1 / (D_G_z2 + 1e-10),
                                               global_step)
                        self.writer.add_histogram('D_real_logits', d_real.cpu().data,
                                                  global_step, bins='sturges')
                        self.writer.add_histogram('D_fake_logits', d_fake.cpu().data,
                                                  global_step, bins='sturges')
                        self.writer.add_histogram('D_fake__logits', d_fake_.cpu().data,
                                                  global_step, bins='sturges')
                    self.writer.add_scalar('d_noise_std', d_noise_std,
                                           global_step)
                    self.writer.add_scalar('D_real', d_real_loss_v,
                                           global_step)
                    self.writer.add_scalar('D_fake', d_fake_loss_v,
                                           global_step)
                    self.writer.add_scalar('G_adv', g_adv_loss_v,
                                           global_step)
                    self.writer.add_histogram('Gz', Genh.cpu().data.numpy(),
                                              global_step, bins='sturges')
                    self.writer.add_histogram('clean', clean.cpu().data.numpy(),
                                              global_step, bins='sturges')
                    self.writer.add_histogram('noisy', noisy.cpu().data.numpy(),
                                              global_step, bins='sturges')
                    self.writer.add_scalar('G_noisy_pad_size', 
                                           pad_size,
                                           global_step)
                    # histogram of intermediate activations from d_real and
                    # d_fake
                    for k, v in d_real_iact.items():
                        self.writer.add_histogram(k, v.cpu().data, global_step,
                                                  bins='sturges')
                    for k, v in d_fake_iact.items():
                        self.writer.add_histogram(k, v.cpu().data, global_step,
                                                  bins='sturges')
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
                                                       global_step)
                                total_GW_norm += W_norm
                        self.writer.add_scalar('{}_Wnorm'.format(total_name),
                                               total_GW_norm,
                                               global_step)
                    model_weights_norm(self.G, 'Gtotal')
                    model_weights_norm(self.D, 'Dtotal')
                    # Plot G skip connection parameters
                    for skip_idx, skip_conn in self.G.skips.items():
                        alpha = skip_conn['alpha'].cpu().data
                        self.writer.add_scalar('skip_{}_a-norm'.format(skip_idx),
                                               torch.norm(alpha),
                                               global_step)
                        self.writer.add_histogram('skip_{}_a'.format(skip_idx),
                                                  alpha,
                                                  global_step, bins='sturges')
                    canvas_w = self.G(noisy_samples, z=z_sample)
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
                                                   '{}.wav'.format(global_step,
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
                    # save model
                    self.save(self.save_path, global_step)
                global_step += 1

            if va_dloader is not None:
                #pesqs, mpesq = self.evaluate(opts, va_dloader, log_freq)
                pesqs, npesqs, \
                mpesq, mnpesq, \
                ssnrs, nssnrs, \
                mssnr, mnssnr = self.evaluate(opts, va_dloader, 
                                              log_freq, do_noisy=True)
                print('mean noisyPESQ: ', mnpesq)
                print('mean GenhPESQ: ', mpesq)
                self.writer.add_scalar('noisyPESQ', mnpesq, epoch)
                self.writer.add_scalar('noisySSNR', mnssnr, epoch)
                self.writer.add_scalar('GenhPESQ', mpesq, epoch)
                self.writer.add_scalar('GenhSSNR', mssnr, epoch)
                #self.writer.add_histogram('noisyPESQ', npesqs,
                #                          epoch, bins='sturges')
                #self.writer.add_histogram('GenhPESQ', pesqs,
                #                          epoch, bins='sturges')


    def evaluate(self, opts, dloader, log_freq, do_noisy=False,
                 max_samples=100):
        """ Objective evaluation with PESQ and SSNR """
        self.G.eval()
        self.D.eval()
        beg_t = timeit.default_timer()
        pesqs = []
        ssnrs = []
        if do_noisy:
            npesqs = []
            nssnrs = []
        total_s = 0
        timings = []
        # going over dataset ONCE
        for bidx, batch in enumerate(dloader, start=1):
            sample = batch
            if len(sample) == 2:
                clean, noisy = batch
            elif len(sample) == 3:
                clean, noisy, pesq = batch
            else:
                clean, noisy, pesq, ssnr = batch
            clean = Variable(clean, volatile=True)
            noisy = Variable(noisy.unsqueeze(1), volatile=True)
            if self.do_cuda:
                clean = clean.cuda()
                noisy = noisy.cuda()
            Genh = self.G(noisy).squeeze(1)
            clean_npy = clean.cpu().data.numpy()
            if do_noisy:
                noisy_npy = noisy.cpu().data.numpy()
            Genh_npy = Genh.cpu().data.numpy()
            for sidx in range(Genh.size(0)):
                clean_utt = denormalize_wave_minmax(clean_npy[sidx]).astype(np.int16)
                clean_utt = clean_utt.reshape(-1)
                clean_utt = de_emphasize(clean_utt, self.preemph)
                Genh_utt = denormalize_wave_minmax(Genh_npy[sidx]).astype(np.int16)
                Genh_utt = Genh_utt.reshape(-1)
                Genh_utt = de_emphasize(Genh_utt, self.preemph)
                # compute PESQ per file
                pesq = PESQ(clean_utt, Genh_utt)
                if 'error' in pesq:
                    print('Skipping error')
                    continue
                pesq = float(pesq)
                pesqs.append(pesq)
                snr_mean, segsnr_mean = SSNR(clean_utt, Genh_utt)
                segsnr_mean = float(segsnr_mean)
                ssnrs.append(segsnr_mean)
                print('Genh sample {} > PESQ: {:.3f}, SSNR: {:.3f} dB'
                      ''.format(total_s, pesq, segsnr_mean))
                if do_noisy:
                    # noisy PESQ too
                    noisy_utt = denormalize_wave_minmax(noisy_npy[sidx]).astype(np.int16)
                    noisy_utt = noisy_utt.reshape(-1)
                    noisy_utt = de_emphasize(noisy_utt, self.preemph)
                    npesq = PESQ(clean_utt, noisy_utt)
                    npesq = float(npesq)
                    npesqs.append(npesq)
                    nsnr_mean, nsegsnr_mean = SSNR(clean_utt, noisy_utt)
                    nsegsnr_mean = float(nsegsnr_mean)
                    nssnrs.append(nsegsnr_mean)
                    print('Noisy sample {} > PESQ: {:.3f}, SSNR: {:.3f} dB'
                          ''.format(total_s, npesq, nsegsnr_mean))
                # Segmental SNR
                total_s += 1
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                print('Mean pesq computation time: {}'
                      's'.format(np.mean(timings)))
                beg_t = timeit.default_timer()
                #print('{} PESQ: {}'.format(sidx, pesq))
                #wavfile.write('{}_clean_test.wav'.format(sidx), 16000,
                #              clean_utt)
                #wavfile.write('{}_enh_test.wav'.format(sidx), 16000,
                #              Genh_utt)
            #if bidx % log_freq == 0 or bidx >= len(dloader):
            #    print('EVAL Batch {}/{} mPESQ: {:.4f}'
            #          ''.format(bidx,
            #                    len(dloader),
            #                    np.mean(pesqs)))
            if total_s >= max_samples:
                break
        return np.array(pesqs), np.array(npesqs), np.mean(pesqs), \
               np.mean(npesqs), np.array(ssnrs), \
               np.array(nssnrs), np.mean(ssnrs), np.mean(nssnrs)
