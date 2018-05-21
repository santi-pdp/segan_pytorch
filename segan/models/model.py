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
                                 skip=True,
                                 dec_activations=self.g_dec_act,
                                 bias=True,
                                 skip_init='one',
                                 dec_kwidth=opts.kwidth,
                                 skip_type=opts.skip_type,
                                 skip_merge=opts.skip_merge)
        else:
            self.G = generator

        self.G.apply(weights_init)
        print('Generator: ', self.G)

        self.d_enc_fmaps = opts.d_enc_fmaps
        if discriminator is None:
            self.D = Discriminator(2, self.d_enc_fmaps, opts.kwidth,
                                   nn.LeakyReLU(0.3), 
                                   bnorm=True, pooling=opts.pooling_size, 
                                   pool_type='conv',
                                   pool_size=opts.D_pool_size)
        else:
            self.D = discriminator
        self.D.apply(weights_init)
        print('Discriminator: ', self.D)
        if self.do_cuda:
            self.D.cuda()
            self.G.cuda()
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))

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
            if pad  > 0:
                x[0, 0] = torch.cat((inwav[0, 0, beg_i:beg_i + length],
                                    torch.zeros(pad)), dim=0)
            else:
                x[0, 0] = inwav[0, 0, beg_i:beg_i + length]
            x = torch.FloatTensor(x)
            #canvas_w, hall = self.G(x, z=z, ret_hid=True)
            canvas_w, hall = self.infer_G(x, z=z, ret_hid=True)
            nums = []
            for k in hall.keys():
                if 'enc' in k and 'zc' not in k:
                    nums.append(int(k.split('_')[1]))
            g_c = hall['enc_{}'.format(max(nums))]
            if z is None:
                # if z was created inside G as first inference
                z = self.G.z
            if pad > 0:
                canvas_w = canvas_w[0, 0, :-pad]
            canvas_w = canvas_w.data.numpy().squeeze()
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        # de-emph
        c_res = de_emphasize(c_res, self.preemph)
        return c_res, g_c

    def discriminate(self, cwav, nwav):
        self.D.eval()
        d_in = torch.cat((cwav, nwav), dim=1)
        d_veredict, _ = self.D(d_in)
        return d_veredict

    def infer_G(self, nwav, cwav=None, z=None, ret_hid=False):
        Genh = self.G(nwav, z=z, ret_hid=ret_hid)
        return Genh

    def infer_D(self, x_, ref):
        D_in = torch.cat((x_, ref), dim=1)
        return self.D(D_in)

    def gen_train_samples(self, clean_samples, noisy_samples, z_sample, 
                          global_step=None):
        canvas_w = self.infer_G(noisy_samples, clean_samples, z=z_sample)
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


    def train(self, opts, dloader, criterion, l1_init, l1_dec_step,
              l1_dec_epoch, log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        Gopt = optim.RMSprop(self.G.parameters(), lr=opts.g_lr)
        Dopt = optim.RMSprop(self.D.parameters(), lr=opts.d_lr)

        # attach opts to models so that they are saved altogether in ckpts
        self.G.optim = Gopt
        self.D.optim = Dopt

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
                #Genh = self.G(noisy)
                Genh = self.infer_G(noisy, clean)
                lab = Variable(label)
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
                lab = Variable(label.fill_(0))
                d_fake_loss = criterion(d_fake.view(-1), lab)
                d_fake_loss.backward()
                total_d_fake_loss += d_fake_loss
                Dopt.step()

                d_loss = d_fake_loss + d_real_loss 

                # (3) G real update
                Gopt.zero_grad()
                lab = Variable(label.fill_(1))
                #d_fake_, _ = self.D(torch.cat((Genh, noisy), dim=1))
                d_fake_, _ = self.infer_D(Genh, noisy)
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
                    #canvas_w = self.G(noisy_samples, z=z_sample)
                    self.gen_train_samples(clean_samples, noisy_samples,
                                           z_sample,
                                           global_step=global_step)
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
                self.writer.add_scalar('Genh-val_obj',
                                       val_obj, epoch)
                if val_obj > best_val_obj:
                    print('Val obj (COVL + SSNR) improved '
                          '{} -> {}'.format(best_val_obj,
                                            val_obj))
                    best_val_obj = val_obj
                    patience = opts.patience
                    # save models
                    self.G.save(self.save_path, global_step, True)
                    self.D.save(self.save_path, global_step, True)
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
                self.G.save(self.save_path, global_step)
                self.D.save(self.save_path, global_step)


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
            #Genh = self.G(noisy).squeeze(1)
            Genh = self.infer_G(noisy).squeeze(1)
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


class VCSEGAN(SEGAN):
    def __init__(self, opts, name='VCSEGAN',
                 generator=None, discriminator=None):
        super().__init__(opts, name=name, generator=generator,
                         discriminator=discriminator)

    def infer_G(self, nwav, cwav, z=None, ret_hid=False):
        # include tsteps in G interface to change num steps out than in
        N_div = self.pooling ** len(self.g_enc_fmaps)
        Genh = self.G(nwav,
                      dec_steps=int(np.ceil(cwav.size(2)/N_div)),
                      z=z, ret_hid=ret_hid)
        if ret_hid:
            Genh, hid = Genh
        if Genh.size(2) > cwav.size(2):
            # trim part sobrant
            Genh = Genh[:, :, :cwav.size(2)]
        if ret_hid:
            return Genh, hid
        else:
            return Genh

    def infer_D(self, x_, ref):
        x_len = x_.size(2)
        ref_len = ref.size(2)
        bsz = ref.size(0)
        kdim = ref.size(1)
        assert bsz == x_.size(0), x.size(0)
        assert kdim == x_.size(1), x.size(1)
        if x_len > ref_len:
            pad = torch.zeros(bsz, kdim, x_len - ref_len)
            if self.do_cuda:
                pad = pad.cuda()
            ref = torch.cat((ref, pad),
                            dim=2)
        elif x_len < ref_len:
            pad = torch.zeros(bsz, kdim, ref_len - x_len)
            if self.do_cuda:
                pad = pad.cuda()
            x_ = torch.cat((x_, pad),
                           dim=2)
        #print('x_ size: ', x_.size())
        #print('ref size: ', ref.size())
        D_in = torch.cat((x_, ref), dim=1)
        return self.D(D_in)

    def gen_train_samples(self, clean_samples, noisy_samples, z_sample,
                          global_step=None):
        canvas_w, hid = self.infer_G(noisy_samples, clean_samples, z=z_sample,
                                     ret_hid=True)
        att = hid['att']
        att = att.unsqueeze(1) * 1000
        x = vutils.make_grid(att)
        #print('Gen att size: ', att.size())
        self.writer.add_image('latent_att', x, global_step)
        if self.G.skip:
            for n in range(len(self.G.gen_dec)):
                curr_attn = hid['att_{}'.format(n)]
                x = vutils.make_grid(curr_attn)
                self.writer.add_image('att_{}'.format(n),
                                      x, global_step)

