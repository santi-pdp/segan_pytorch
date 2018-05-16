import torch
import torch.nn as nn
from random import shuffle
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from ..datasets import *
from ..utils import *
from .ops import *
from scipy.io import wavfile
import multiprocessing as mp
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
from torch import autograd


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print('Initialzing weight to 0.0, 0.02')
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            print('bias to 0 for module: ', m)
            m.bias.data.fill_(0)
    if classname.find('Linear') != -1:
        print('Initializing FC weight to xavier uniform')
        nn.init.xavier_uniform(m.weight.data)
    #elif classname.find('BatchNorm') != -1:
    #    m.weight.data.normal_(1.0, 0.02)

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
        #self.g_enc_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
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
        self.G = Generator1D(1, 
                             self.g_enc_fmaps, 
                             opts.kwidth,
                             self.g_enc_act,
                             lnorm=False, dropout=0.,
                             pooling=opts.pooling_size,
                             z_dim=self.g_enc_fmaps[-1],
                             z_all=False,
                             cuda=opts.cuda,
                             skip=True,
                             skip_blacklist=[],
                             dec_activations=self.g_dec_act,
                             bias=True,
                             aal_out=False,
                             wd=0., skip_init='one',
                             skip_dropout=0,
                             no_tanh=False,
                             rnn_core=False,
                             linterp=opts.linterp,
                             mlpconv=False,
                             dec_kwidth=31,
                             subtract_mean=False,
                             no_z=False,
                             skip_type=opts.skip_type,
                             num_spks=None,
                             skip_merge=opts.skip_merge)
        self.G.apply(weights_init)
        print('Generator: ', self.G)

        self.d_enc_fmaps = opts.d_enc_fmaps
        if opts.disc_type == 'vbnd':
            self.D = VBNDiscriminator(2, self.d_enc_fmaps, opts.kwidth,
                                      nn.LeakyReLU(0.3), 
                                      pooling=opts.pooling_size, SND=opts.SND,
                                      pool_size=opts.D_pool_size,
                                      cuda=opts.cuda)

        else:
            self.D = Discriminator(2, self.d_enc_fmaps, opts.kwidth,
                                   nn.LeakyReLU(0.3), 
                                   bnorm=True,
                                   pooling=opts.pooling_size, 
                                   SND=opts.SND,
                                   pool_type='conv',
                                   dropout=0,
                                   Genc=None,
                                   pool_size=opts.D_pool_size,
                                   num_spks=None)
        self.D.apply(weights_init)
        print('Discriminator: ', self.D)
        if self.do_cuda:
            self.D.cuda()
            self.G.cuda()
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))

    def load_raw_weights(self, raw_weights_dir):
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
        self.G.eval()
        N = 16384
        x = np.zeros((1, 1, N))
        c_res = None
        for beg_i in range(0, inwav.shape[2], N):
            if inwav.shape[2] - beg_i < N:
                length = inwav.shape[2] - beg_i
                pad = N - length
            else:
                length = N
                pad = 0
            print('padding:',pad)
            if pad  > 0:
                x[0, 0] = torch.cat((inwav[0, 0, beg_i:beg_i + length],
                                    torch.zeros(pad)), dim=0)
            else:
                x[0, 0] = inwav[0, 0, beg_i:beg_i + length]
            print('Cleaning chunk {} -> {}'.format(beg_i, beg_i + length))
            x = torch.FloatTensor(x)
            print('x size: ', x.size())
            canvas_w, hall = self.G(x, z=z, ret_hid=True)
            print(list(hall.keys()))
            g_c = hall['enc_10']
            if z is None:
                # if z was created inside G as first inference
                z = self.G.z
            if pad > 0:
                print('Removing padding of {} samples'.format(pad))
                canvas_w = canvas_w[0, 0, :-pad]
            canvas_w = canvas_w.data.numpy().squeeze()
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        print('inwav size: ', inwav.size())
        # de-emph
        print('c_res shape: ', c_res.shape)
        c_res = de_emphasize(c_res, self.preemph)
        print('c_res shape de_emph: ', c_res.shape)
        return c_res, g_c
        """
        y = self.G(inwav)
        if self.preemph > 0:
            print('De-emphasis of : ', self.preemph)
            out_y = []
            for b_i in range(y.size(0)):
                print('de_emph y[{}, 0, :] with size {}'.format(b_i,
                                                                y[b_i, 0,
                                                                  :].size()))
                out_yy = de_emphasize(y[b_i, 0, :].cpu().data.numpy(),
                                      self.preemph)
                out_y.append(torch.FloatTensor(out_yy))
            out_y = Variable(torch.cat(out_y, dim=0).unsqueeze(1))
            y = out_y
        print('y size: ', y.size())
        #m_noisy = de_emphasize(noisy_samples[m,
        return y
        """

    def discriminate(self, cwav, nwav):
        self.D.eval()
        d_in = torch.cat((cwav, nwav), dim=1)
        d_veredict, _ = self.D(d_in)
        return d_veredict


    def train(self, opts, dloader, criterion, l1_init, l1_dec_step,
              l1_dec_epoch, log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
        Dopt = optim.RMSprop(self.D.parameters(), lr=opts.d_lr)

        num_batches = len(dloader) 


        l1_weight = l1_init
        global_step = 1
        timings = []
        evals = {}
        noisy_evals = {}
        noisy_samples = None
        clean_samples = None
        z_sample = None
        patience = opts.patience
        best_val_obj = 0
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
                if len(sample) == 3:
                    uttname, clean, noisy = batch
                else:
                    raise ValueError('Returned {} elements per '
                                     'sample?'.format(len(sample)))
                clean = Variable(clean.unsqueeze(1))
                noisy = Variable(noisy.unsqueeze(1))
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
                Genh = self.G(noisy)
                for d_i in range(opts.d_updates):
                    lab = Variable(label)
                    D_in = torch.cat((clean, noisy), dim=1)
                    d_real, _ = self.D(D_in)
                    d_real_loss = criterion(d_real.view(-1), lab)
                    d_real_loss.backward()
                    total_d_real_loss += d_real_loss
                    
                    # (2) D fake update
                    D_fake_in = torch.cat((Genh.detach(), noisy), dim=1)
                    d_fake, _ = self.D(D_fake_in)
                    # Make fake objective
                    lab = Variable(label.fill_(0))
                    d_fake_loss = criterion(d_fake.view(-1), lab)
                    d_fake_loss.backward()
                    total_d_fake_loss += d_fake_loss
                    Dopt.step()

                d_loss = (d_fake_loss + d_real_loss) / opts.d_updates

                # (3) G real update
                Gopt.zero_grad()
                lab = Variable(label.fill_(1))
                d_fake_, _ = self.D(torch.cat((Genh, noisy), dim=1))
                g_adv_loss = criterion(d_fake_.view(-1), lab)
                g_l1_loss = l1_weight * F.l1_loss(Genh, clean)
                g_loss = g_adv_loss + g_l1_loss
                g_loss.backward()
                Gopt.step()
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if z_sample is None:
                    # capture sample now that we know shape after first
                    # inference
                    z_sample = self.G.z[:20, :, :].contiguous()
                    print('z_sample size: ', z_sample.size())
                    if self.do_cuda:
                        z_sample = z_sample.cuda()
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    d_real_loss_v = d_real_loss.cpu().data[0]
                    d_fake_loss_v = d_fake_loss.cpu().data[0]
                    g_adv_loss_v = g_adv_loss.cpu().data[0]
                    g_l1_loss_v = g_l1_loss.cpu().data[0]
                    log = '(Iter {}) Batch {}/{} (Epoch {}) d_real:{:.4f}, ' \
                          'd_fake:{:.4f}, '.format(global_step, bidx,
                                                   len(dloader), epoch,
                                                   d_real_loss_v,
                                                   d_fake_loss_v)
                    log += 'g_adv:{:.4f}, g_l1:{:.4f} ' \
                           'l1_w: {:.2f}, btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(g_adv_loss_v,
                                     g_l1_loss_v, l1_weight, timings[-1],
                                     np.mean(timings))
                    print(log)
                    self.writer.add_scalar('D_real', d_real_loss_v,
                                           global_step)
                    self.writer.add_scalar('D_fake', d_fake_loss_v,
                                           global_step)
                    self.writer.add_scalar('G_adv', g_adv_loss_v,
                                           global_step)
                    self.writer.add_scalar('G_l1', g_l1_loss_v,
                                           global_step)
                    self.writer.add_histogram('D_fake__hist', d_fake_.cpu().data,
                                              global_step, bins='sturges')
                    self.writer.add_histogram('D_fake_hist', d_fake.cpu().data,
                                              global_step, bins='sturges')
                    self.writer.add_histogram('D_real_hist', d_real.cpu().data,
                                              global_step, bins='sturges')
                    self.writer.add_histogram('Gz', Genh.cpu().data,
                                              global_step, bins='sturges')
                    self.writer.add_histogram('clean', clean.cpu().data,
                                              global_step, bins='sturges')
                    self.writer.add_histogram('noisy', noisy.cpu().data,
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
                global_step += 1

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
                if val_obj > best_val_obj:
                    print('Val obj (COVL + SSNR) improved '
                          '{} -> {}'.format(best_val_obj,
                                            val_obj))
                    best_val_obj = val_obj
                    patience = opts.patience
                    # save model
                    self.save(self.save_path, global_step, True)
                else:
                    patience -= 1
                    print('Val loss did not improve. Patience'
                          '{}/{}'.format(patience,
                                        opts.patience))
                    if patience <= 0:
                        print('STOPPING SEGAN TRAIN: OUT OF PATIENCE.')
                        break
                
            else:
                # save model
                self.save(self.save_path, global_step)



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
        # going over dataset ONCE
        for bidx, batch in enumerate(dloader, start=1):
            sample = batch
            if len(sample) == 3:
                uttname, clean, noisy = batch
            else:
                raise ValueError('Returned {} elements per '
                                 'sample?'.format(len(sample)))
            clean = Variable(clean, volatile=True)
            noisy = Variable(noisy.unsqueeze(1), volatile=True)
            if self.do_cuda:
                clean = clean.cuda()
                noisy = noisy.cuda()
            Genh = self.G(noisy).squeeze(1)
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
            print('Time to process eval: {} s'.format(end_t - beg_t))
            if bidx >= max_samples:
                break
            """
            for sidx in range(Genh.size(0)):
                clean_utt = clean_npy[sidx].reshape(-1)
                #clean_utt = de_emphasize(clean_utt, self.preemph)
                Genh_utt = Genh_npy[sidx].reshape(-1)
                try:
                    #Genh_utt = de_emphasize(Genh_utt, self.preemph)
                    csig, cbak, covl, pesq, ssnr = CompositeEval(clean_utt,
                                                                 Genh_utt, 
                                                                 True)
                except ValueError:
                    continue
                evals['pesq'].append(pesq)
                evals['ssnr'].append(ssnr)
                evals['csig'].append(csig)
                evals['cbak'].append(cbak)
                evals['covl'].append(covl)
                #print('Genh sample {} > PESQ: {:.3f}, SSNR: {:.3f} dB'
                #      ''.format(total_s, pesq, segsnr_mean))
                if do_noisy:
                    # noisy PESQ too
                    noisy_utt = noisy_npy[sidx].reshape(-1)
                    #noisy_utt = de_emphasize(noisy_utt, self.preemph)
                    csig, cbak, covl, pesq, ssnr = CompositeEval(clean_utt,
                                                                 noisy_utt,
                                                                 True)
                    noisy_evals['pesq'].append(pesq)
                    noisy_evals['ssnr'].append(ssnr)
                    noisy_evals['csig'].append(csig)
                    noisy_evals['cbak'].append(cbak)
                    noisy_evals['covl'].append(covl)
                    #print('Noisy sample {} > PESQ: {:.3f}, SSNR: {:.3f} dB'
                    #      ''.format(total_s, npesq, nsegsnr_mean))
                # Segmental SNR
                total_s += 1
                #end_t = timeit.default_timer()
                #timings.append(end_t - beg_t)
                #print('Mean pesq computation time: {}'
                #      's'.format(np.mean(timings)))
                #beg_t = timeit.default_timer()
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
            #if total_s >= max_samples:
            #    break
            end_t = timeit.default_timer()
            #print('evals len: ', len(evals))
            print('Time to process eval: {} s'.format(end_t - beg_t))
            break
        """
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
        self.pooling_size=opts.pooling_size
        self.no_eval = opts.no_eval
        self.d_real_weight = opts.d_real_weight
        self.g_weight = opts.g_weight
        self.d_fake_weight = opts.d_fake_weight
        #self.f0_evaluator = F0Evaluator(cuda=opts.cuda)
        self.f0_evaluator = None
        self.noise_dec_step = opts.noise_dec_step
        self.saver = Saver(model=self, save_path=opts.save_path, 
                           max_ckpts=opts.max_ckpts)
        # add misalignment fake signal in stereoD
        self.misalign_stereo = opts.misalign_stereo
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
        self.g_onehot = opts.g_onehot
        if opts.g_onehot:
            g_num_spks = self.num_spks
        else:
            g_num_spks = None
        # Build G and D
        self.G = Generator1D(1, self.g_enc_fmaps, opts.kwidth,
                             self.g_enc_act,
                             lnorm=opts.g_bnorm, dropout=opts.g_dropout, 
                             pooling=self.pooling_size, z_dim=opts.z_dim,
                             z_all=opts.z_all,
                             cuda=opts.cuda,
                             skip=opts.skip,
                             skip_blacklist=opts.skip_blacklist,
                             dec_activations=self.g_dec_act,
                             bias=opts.g_bias,
                             aal=opts.g_aal,
                             aal_out=opts.g_aal_out, wd=0.,
                             skip_init=opts.skip_init,
                             skip_dropout=opts.skip_dropout,
                             no_tanh=opts.no_tanh,
                             rnn_core=opts.g_rnn_core,
                             linterp=opts.linterp,
                             mlpconv=opts.g_mlpconv,
                             dec_kwidth=opts.dec_kwidth,
                             subtract_mean=opts.g_subtract_mean,
                             no_z=opts.no_z,
                             skip_type=opts.skip_type,
                             num_spks=g_num_spks)
                             
        if not opts.no_winit:
            self.G.apply(weights_init)
        print('Generator: ', self.G)
        if opts.d_act == 'prelu':
            self.d_act = [nn.PReLU(fmaps) for fmaps in opts.d_enc_fmaps]
        elif opts.d_act == 'lrelu':
            self.d_act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('Unrecognied D activation: ', self.d_act)
        self.d_enc_fmaps = opts.d_enc_fmaps
        if self.BID:
            self.D = BiDiscriminator(self.d_enc_fmaps, opts.kwidth,
                                     self.d_act,
                                     bnorm=opts.d_bnorm,
                                     pooling=self.pooling_size, SND=opts.SND,
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
                                   self.d_act,
                                   bnorm=opts.d_bnorm,
                                   pooling=self.pooling_size, SND=opts.SND,
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


    def generate(self, inwav, ret_hid=False, spkid=None):
        return self.G(inwav, ret_hid=ret_hid, spkid=spkid)

    def discriminate(self, inwav):
        return self.D(inwav)

    def train(self, opts, dloader, criterion, 
              log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        Gopt, g_sched = make_optimizer(self.g_optim, self.G.parameters(),
                                       opts.g_lr, opts.g_step_lr,
                                       opts.g_lr_gamma,
                                       opts.beta_1,
                                       opts.wd)
        Dopt, d_sched = make_optimizer(self.d_optim, self.D.parameters(),
                                       opts.d_lr, opts.d_step_lr,
                                       opts.d_lr_gamma,
                                       opts.beta_1, 
                                       opts.wd)

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
                if len(sample) == 3:
                    uttname, clean, noisy = batch
                elif len(sample) == 4:
                    if self.num_spks is None:
                        raise ValueError('Need num_spks active in SilentSEGAN'\
                                         ' when delivering spkid in dataloader')
                    uttname, clean, noisy, spkid = batch
                    spkid = Variable(spkid).view(-1)
                    if self.do_cuda:
                        spkid = spkid.cuda()
                else:
                    raise ValueError('Returned {} elements per '
                                     'sample?'.format(len(sample)))
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
                    if self.g_onehot:
                        spkid_samples = spkid[:20]
                    else:
                        spkid_samples = None
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
                g_spkid = None
                if self.g_onehot:
                    g_spkid = spkid
                Genh = self.G(noisy, spkid=g_spkid)
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

                if self.stereo_D and self.misalign_stereo:
                    # add clean, misaligned_clean pair as fake
                    idxs = list(range(clean.size(0)))
                    shuffle(idxs)
                    sh_clean = clean[idxs]
                    #D_fakeagn_in = torch.cat((sh_clean, noisy), dim=1)
                    D_fakeagn_in = torch.cat((clean, sh_clean), dim=1)
                    d_fakeagn, d_fakeagn_iact = self.D(D_fakeagn_in)
                    if spkid is not None:
                        d_fakeagn = d_fakeagn[:, :1].contiguous()
                    d_fake_loss += self.d_fake_weight * criterion(d_fakeagn,
                                                                  lab)
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
                        skip_ps = dict(skip_conn['alpha'].named_parameters())
                        for k, v in skip_ps.items():
                            self.writer.add_scalar('{}-{}_norm'.format(k, skip_idx),
                                                   torch.norm(v),
                                                   global_step)
                            self.writer.add_histogram('{}-{}'.format(k, skip_idx),
                                                      v,
                                                      global_step, bins='sturges')
                    canvas_w = self.G(noisy_samples, z=z_sample, spkid=spkid_samples)
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

            if va_dloader is not None and not self.no_eval:
                #pesqs, mpesq = self.evaluate(opts, va_dloader, log_freq)
                # need to trim to 10 samples cause it is slow process
                D_xs, D_G_zs = self.evaluate(opts, va_dloader, criterion,
                                             epoch,
                                             max_samples=10)


    def evaluate(self, opts, dloader, criterion, epoch, max_samples=100):
        """ Evaluate D_x, G_z1 and G_z2 with validation/test data """
        self.G.eval()
        self.D.eval()
        beg_t = timeit.default_timer()
        total_s = 0
        timings = []
        D_xs = []
        spkid = None
        D_G_zs = []
        G_fake_losses = []
        D_real_losses = []
        G_spk_losses = []
        D_spk_losses = []
        # store eval results from F0Evaluator
        klds = []
        maes = []
        accs = []
        beg_eval_t = timeit.default_timer()
        # going over dataset ONCE
        for bidx, batch in enumerate(dloader, start=1):
            sample = batch
            if len(sample) == 3:
                uttname, clean, noisy = batch
            elif len(sample) == 4:
                if self.num_spks is None:
                    raise ValueError('Need num_spks active in SilentSEGAN'\
                                     ' when delivering spkid in dataloader')
                uttname, clean, noisy, spkid = batch
                spkid = Variable(spkid).view(-1)
                if self.do_cuda:
                    spkid = spkid.cuda()
            clean_npy = clean.numpy()
            clean = Variable(clean.unsqueeze(1), volatile=True)
            noisy = Variable(noisy.unsqueeze(1), volatile=True)
            if self.do_cuda:
                clean = clean.cuda()
                noisy = noisy.cuda()
            # prepare lab to compute eval loss only with true target
            lab = Variable(torch.ones(clean.size(0), 1))
            if self.do_cuda:
                lab = lab.cuda()
            g_spkid = None
            if self.g_onehot:
                g_spkid = spkid
            # forward through G
            Genh = self.G(noisy, spkid=g_spkid)
            Genh_npy = Genh.view(clean.size(0), -1).cpu().data.numpy()
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
            if self.f0_evaluator is not None:
                kld, mae, acc = self.f0_evaluator(Genh_npy, clean_npy)
                klds.append(kld) 
                maes.append(mae)
                accs.append(acc)
            end_eval_t = timeit.default_timer()
            timings.append(end_eval_t - beg_eval_t)
            print('Eval batch {}/{} computed in {} s, mbtime: {} '
                  's'.format(bidx, len(dloader), timings[-1],
                            np.mean(timings)))
            if bidx >= max_samples:
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
        if self.f0_evaluator is not None:
            self.writer.add_scalar('meanEval-KLD', np.mean(klds),
                                   epoch)
            self.writer.add_scalar('meanEval-MAE_Hz', np.mean(maes),
                                   epoch)
            self.writer.add_scalar('meanEval-ACC_norm', np.mean(accs),
                                   epoch)
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


class WSilentSEGAN(SilentSEGAN):
    """ WGAN-GP """

    def __init__(self, opts, name='WSilentSEGAN'):
        super(WSilentSEGAN, self).__init__(opts, name)
        # lambda factor for GP
        self.lbd = opts.lbd
        self.critic_iters = opts.critic_iters

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        bsz = real_data.size(0)
        alpha = torch.rand(bsz, 1)
        alpha = alpha.expand(bsz, real_data.nelement() // bsz).contiguous()
        alpha = alpha.view(real_data.size())
        #print('alpha size: ', alpha.size())
        #print('real_data size: ', real_data.size())
        #print('fake_data size: ', fake_data.size())
        alpha = alpha.cuda() if self.do_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if self.do_cuda:
            interpolates = interpolates.cuda()

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
        gradient_p = torch.mean((1. - torch.sqrt(1e-8 + \
                                                 torch.sum(gr ** 2, \
                                                           dim=1))) ** 2)
                                 
        #gradient_p = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lbd
        return gradient_p * self.lbd

    def sample_dloader(self, dloader):
        sample = next(dloader.__iter__())
        batch = sample

        if len(sample) == 3:
            uttname, clean, noisy = batch
            spkid = None
        elif len(sample) == 4:
            if self.num_spks is None:
                raise ValueError('Need num_spks active in SilentSEGAN'\
                                 ' when delivering spkid in dataloader')
            uttname, clean, noisy, spkid = batch
            spkid = Variable(spkid).view(-1)
            if self.do_cuda:
                spkid = spkid.cuda()
        clean = Variable(clean.unsqueeze(1))
        noisy = Variable(noisy.unsqueeze(1))
        if self.do_cuda:
            clean = clean.cuda()
            noisy = noisy.cuda()
        return uttname, clean, noisy, spkid

    def train(self, opts, dloader, criterion, 
              log_freq, va_dloader=None):

        """ Train the WSEGAN """
        Gopt, g_sched = make_optimizer(self.g_optim, self.G.parameters(),
                                       opts.g_lr, opts.g_step_lr,
                                       opts.g_lr_gamma)
        Dopt, d_sched = make_optimizer(self.d_optim, self.D.parameters(),
                                       opts.d_lr, opts.d_step_lr,
                                       opts.d_lr_gamma)


        num_batches = len(dloader) 

        self.G.train()
        self.D.train()

        global_step = 1
        timings = []
        noisy_samples = None
        clean_samples = None
        z_sample = None
        D_x = None
        D_G_z1 = None
        D_G_z2 = None
        spkid = None
        l1_weight = opts.l1_weight
        # TODO: DO OPT UPDATE????
        # -------------------------------
        # OPTIMIZER UPDATE
        # -------------------------------
        #if g_sched is not None:
        #    G_lr = Gopt.param_groups[0]['lr']
            # WARNING: This does not ensure we will have g_min_lr
            # BUT we will have a maximum update to downgrade lr prior
            # to certain minima
        #    if G_lr > opts.g_min_lr:
        #        g_sched.step()
        #if d_sched is not None:
        #    D_lr = Dopt.param_groups[0]['lr']
            # WARNING: This does not ensure we will have g_min_lr
            # BUT we will have a maximum update to downgrade lr prior
            # to certain minima
        #    if D_lr > opts.d_min_lr:
        #        d_sched.step()
        # -------------------------------
        # no epochs in this SEGAN, but ITERS in total
        for iteration in range(1, opts.iters + 1):

            beg_t = timeit.default_timer()
            for p in self.D.parameters(): 
                # reset requires grad
                p.requires_grad = True

            for critic_i in range(1, self.critic_iters + 1):
                # grads
                one = torch.FloatTensor([1])
                mone = one * -1
                if self.do_cuda:
                    one = one.cuda()
                    mone = mone.cuda()
                # sample batch of data
                uttname, clean, noisy, spkid = self.sample_dloader(dloader)
                self.D.zero_grad()

                # shift slightly noisy as input to G
                if opts.max_pad > 0:
                    pad_size = np.random.randint(0, opts.max_pad)
                    left = np.random.rand(1)[0]
                    if left > 0.5:
                        # pad left
                        pad = (pad_size, 0)
                    else:
                        # pad right
                        pad = (0, pad_size)
                    pnoisy = F.pad(noisy, pad, mode='reflect')
                    #pclean = F.pad(clean, pad, mode='reflect')
                    if left:
                        noisy = pnoisy[:, :, :noisy.size(-1)].contiguous()
                        #clean = pclean[:, :, :clean.size(-1)].contiguous()
                    else:
                        noisy = pnoisy[:, :, -noisy.size(-1):].contiguous()
                        #clean = pclean[:, :, -clean.size(-1):].contiguous()
                else:
                    pad_size = 0

                if noisy_samples is None:
                    # capture some inference data
                    noisy_samples = noisy[:20, :, :]
                    clean_samples = clean[:20, :, :]

                # (1) D real update
                assert self.stereo_D
                D_in = torch.cat((clean, noisy), dim=1)
                d_real, d_real_iact= self.D(D_in)
                if spkid is not None:
                    d_spk = d_real[:, 1:].contiguous()
                    d_real = d_real[:, :1].contiguous()
                d_real_loss = d_real.mean()
                #d_real_loss.backward(one)#, retain_graph=True)
                if spkid is not None:
                    d_spkid_loss = F.cross_entropy(d_spk, spkid)
                    #d_spkid_loss.backward()
                
                # (2) D fake update
                Genh = self.G(noisy)
                fake = Genh.detach()
                D_fake_in = torch.cat((fake, noisy), dim=1)
                d_fake, d_fake_iact = self.D(D_fake_in)
                if spkid is not None:
                    d_fake = d_fake[:, :1].contiguous()
                d_fake_loss = d_fake.mean()
                #d_fake_loss.backward(one)


                # train with gradient penalty
                gradient_penalty = self.calc_gradient_penalty(self.D,
                                                              D_in.data, 
                                                              fake.data)
                #gradient_penalty.backward()

                #d_loss = d_fake_loss - d_real_loss + gradient_penalty
                d_loss = -(d_real_loss - d_fake_loss) + gradient_penalty
                total_d_loss = d_loss
                if spkid is not None:
                    total_d_loss += d_spkid_loss
                total_d_loss.backward()
                wasserstein_D = d_real_loss - d_fake_loss
                Dopt.step()


            # (3) G real update
            for p in self.D.parameters():
                p.requires_grad = False # avoid computation
            self.G.zero_grad()
            # sample batch of data
            uttname, clean, noisy, spkid = self.sample_dloader(dloader)
            # infer through G
            Genh = self.G(noisy)
            d_fake__in = torch.cat((Genh, noisy), dim=1)
            d_fake_, d_fake__iact = self.D(d_fake__in)
            if spkid is not None:
                d_spk_ = d_fake_[:, 1:].contiguous()
                d_fake_ = d_fake_[:, :1].contiguous()
            g_adv_loss = d_fake_.mean()
            #g_adv_loss.backward(mone)#, retain_graph=True)
            G_cost = -g_adv_loss
            total_g_cost = G_cost
            if spkid is not None:
                g_spkid_loss = F.cross_entropy(d_spk_, spkid)
                total_g_cost += g_spkid_loss
                #g_spkid_loss.backward()
            if opts.l1_weight > 0:
                g_l1_loss = opts.l1_weight * F.l1_loss(Genh, clean)
                total_g_cost +=g_l1_loss
            total_g_cost.backward()
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
            if iteration % log_freq == 0:
                log = 'Iter {}/{} d_total_loss:{:.4f}, g_loss:{:.4f}, ' \
                      'wass_distance:{:.4f}, ' \
                      ''.format(iteration, opts.iters, total_d_loss.data[0], 
                                G_cost.data[0],
                                wasserstein_D.data[0])
                if self.num_spks is not None:
                    log += 'd_real_spk:{:.4f}, ' \
                           'g_spk:{:.4f}, ' \
                           ''.format(d_spkid_loss.cpu().data[0],
                                     g_spkid_loss.cpu().data[0])
                if opts.l1_weight > 0:
                    log += 'l1_w: {:.1f}, g_l1: ' \
                           '{:.4f},'.format(opts.l1_weight, 
                                            g_l1_loss.cpu().data[0])

                log += ' npad: {:4d},'.format(pad_size)
                log += ' btime: {:.4f} s, mbtime: {:.4f} s, ' \
                       ''.format(timings[-1],
                                 np.mean(timings))
                G_lr = Gopt.param_groups[0]['lr']
                D_lr = Dopt.param_groups[0]['lr']
                log += ', G_curr_lr: ' \
                       '{:.5f}'.format(G_lr)
                log += ', D_curr_lr: ' \
                       '{:.5f}'.format(D_lr)
                print(log)
                if self.num_spks is not None:
                    self.writer.add_scalar('D_spk_loss',
                                           d_spkid_loss.cpu().data[0], global_step)
                    self.writer.add_scalar('G_spk_loss',
                                           g_spkid_loss.cpu().data[0], global_step)
                    self.writer.add_scalar('D_total_loss', total_d_loss.cpu().data,
                                           global_step)
                if opts.l1_weight > 0:
                    self.writer.add_scalar('G_L1_loss', g_l1_loss.cpu().data,
                                           global_step)
                self.writer.add_scalar('G_lr', G_lr, global_step)
                self.writer.add_scalar('D_lr', D_lr, global_step)
                self.writer.add_scalar('D_loss', d_loss.cpu().data,
                                       global_step)
                self.writer.add_scalar('G_noisy_pad_size', 
                                       pad_size,
                                       global_step)
                self.writer.add_scalar('G_loss', G_cost.cpu().data,
                                       global_step)
                self.writer.add_histogram('Gz', Genh.cpu().data,
                                          global_step, bins='sturges')
                self.writer.add_histogram('clean', clean.cpu().data,
                                          global_step, bins='sturges')
                self.writer.add_histogram('noisy', noisy.cpu().data,
                                          global_step, bins='sturges')
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
                    skip_ps = dict(skip_conn['alpha'].named_parameters())
                    for k, v in skip_ps.items():
                        self.writer.add_scalar('{}-{}_norm'.format(k, skip_idx),
                                               torch.norm(v),
                                               global_step)
                        self.writer.add_histogram('{}-{}'.format(k, skip_idx),
                                                  v,
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

        #if va_dloader is not None:
        #    D_xs, D_G_zs = self.evaluate(opts, va_dloader, criterion, epoch)

    def evaluate(self, opts, dloader, criterion, epoch, max_samples=100):
        self.G.eval()
        self.D.eval()
        beg_t = timeit.default_timer()
        total_s = 0
        timings = []
        spkid = None
        D_losses = []
        G_spk_losses = []
        D_spk_losses = []
        # going over dataset ONCE
        for bidx, batch in enumerate(dloader, start=1):
            uttname, clean, noisy, spkid = self.sample_dloader(dloader)
            # forward through G
            Genh = self.G(noisy)
            # forward real and fake through D
            D_real_in = torch.cat((clean, noisy), dim=1)
            d_real, d_real_iact= self.D(D_real_in)
            if spkid is not None:
                d_real_spk = d_real[:, 1:].contiguous()
                d_real = d_real[:, :1].contiguous()
                d_real_spkid_loss = F.cross_entropy(d_real_spk, spkid)
                D_spk_losses.append(d_real_spkid_loss.data[0])
            d_loss = -d_real.mean().cpu().data.numpy()
            D_losses.append(d_loss)

            if spkid is not None:
                # only G loss with fake data in Wass case
                D_fake_in = torch.cat((Genh.detach(), noisy), dim=1)
                d_fake, d_fake_iact= self.D(D_fake_in)
                d_fake_spk = d_fake[:, 1:].contiguous()
                d_fake = d_fake[:, :1].contiguous()
                d_fake_spkid_loss = F.cross_entropy(d_fake_spk, spkid)
                G_spk_losses.append(d_fake_spkid_loss.data[0])
            
            if total_s >= max_samples:
                break
        D_losses = torch.FloatTensor(D_losses)
        if len(G_spk_losses) > 0:
            G_spk_losses = torch.FloatTensor(G_spk_losses)
            D_spk_losses = torch.FloatTensor(D_spk_losses)
        self.writer.add_scalar('meanEval-D_loss',
                               D_losses.mean(), epoch)
        self.writer.add_histogram('Eval-D_loss',
                                   D_losses, epoch,
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

    def __init__(self, opts, name='CycleSEGAN'):
        super(CycleSEGAN, self).__init__(name)
        self.pesq_objective = opts.pesq_objective
        self.preemph = opts.preemph
        self.save_path = opts.save_path
        self.do_cuda = opts.cuda
        self.g_dropout = opts.g_dropout
        self.z_dim = opts.z_dim
        self.g_enc_fmaps = opts.g_enc_fmaps
        #self.g_enc_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
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

        # Build Generators A and B
        self.G_A = Generator1D(1, 
                             self.g_enc_fmaps, 
                             opts.kwidth,
                             self.g_enc_act,
                             lnorm=False, dropout=0.,
                             pooling=opts.pooling_size,
                             z_dim=self.g_enc_fmaps[-1],
                             z_all=False,
                             cuda=opts.cuda,
                             skip=True,
                             skip_blacklist=[],
                             dec_activations=self.g_dec_act,
                             bias=True,
                             aal_out=False,
                             wd=0., skip_init='one',
                             skip_dropout=0,
                             no_tanh=False,
                             rnn_core=False,
                             linterp=opts.linterp,
                             mlpconv=False,
                             dec_kwidth=31,
                             subtract_mean=False,
                             no_z=False,
                             skip_type=opts.skip_type,
                             num_spks=None,
                             skip_merge=opts.skip_merge)
        self.G_A.apply(weights_init)
        self.G_B = Generator1D(1, 
                             self.g_enc_fmaps, 
                             opts.kwidth,
                             self.g_enc_act,
                             lnorm=False, dropout=0.,
                             pooling=opts.pooling_size,
                             z_dim=self.g_enc_fmaps[-1],
                             z_all=False,
                             cuda=opts.cuda,
                             skip=True,
                             skip_blacklist=[],
                             dec_activations=self.g_dec_act,
                             bias=True,
                             aal_out=False,
                             wd=0., skip_init='one',
                             skip_dropout=0,
                             no_tanh=False,
                             rnn_core=False,
                             linterp=opts.linterp,
                             mlpconv=False,
                             dec_kwidth=31,
                             subtract_mean=False,
                             no_z=False,
                             skip_type=opts.skip_type,
                             num_spks=None,
                             skip_merge=opts.skip_merge)
        self.G_B.apply(weights_init)

        self.d_enc_fmaps = opts.d_enc_fmaps
        if opts.disc_type == 'vbnd':
            self.D_A = VBNDiscriminator(2, self.d_enc_fmaps, opts.kwidth,
                                        nn.LeakyReLU(0.3), 
                                        pooling=opts.pooling_size, SND=opts.SND,
                                        pool_size=opts.D_pool_size,
                                        cuda=opts.cuda)
            self.D_B = VBNDiscriminator(2, self.d_enc_fmaps, opts.kwidth,
                                        nn.LeakyReLU(0.3), 
                                        pooling=opts.pooling_size, SND=opts.SND,
                                        pool_size=opts.D_pool_size,
                                        cuda=opts.cuda)

        else:
            self.D_A = Discriminator(2, self.d_enc_fmaps, opts.kwidth,
                                   nn.LeakyReLU(0.3), 
                                   bnorm=opts.d_bnorm,
                                   pooling=2, SND=opts.SND,
                                   pool_type='none',
                                   dropout=0,
                                   Genc=None,
                                   pool_size=opts.D_pool_size,
                                   num_spks=None)
            self.D_B = Discriminator(2, self.d_enc_fmaps, opts.kwidth,
                                   nn.LeakyReLU(0.3), 
                                   bnorm=opts.d_bnorm,
                                   pooling=2, SND=opts.SND,
                                   pool_type='none',
                                   dropout=0,
                                   Genc=None,
                                   pool_size=opts.D_pool_size,
                                   num_spks=None)
        self.D_A.apply(weights_init)
        self.D_B.apply(weights_init)
        print('Discriminator: ', self.D_A)
        if self.do_cuda:
            self.D_A.cuda()
            self.D_B.cuda()
            self.G_A.cuda()
            self.G_B.cuda()
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))


