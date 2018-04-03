import torch
from collections import OrderedDict
import torch.nn as nn
from random import shuffle
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from ..datasets import *
from ..utils import *
from .ops import *
from scipy.io import wavfile
import numpy as np
import timeit
import random
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from .generator import *
from .discriminator import *
from .core import *
from .model import *
import json
import os
from torch import autograd


class F0Predictor(nn.Module):

    def __init__(self, enc_fmaps, enc_acts, kwidth, pooling, lnorm,
                 rnn_size, cuda=False):
        super().__init__()
        bias = True
        self.do_cuda = cuda
        self.rnn_size = rnn_size
        self.enc_odict = OrderedDict()
        # Build Encoder
        for layer_idx, (fmaps, act) in enumerate(zip(enc_fmaps, 
                                                     enc_acts)):
            if layer_idx == 0:
                inp = 1
            else:
                inp = enc_fmaps[layer_idx - 1]
            block = GBlock(inp, fmaps, kwidth, act,
                           padding=None, lnorm=lnorm, 
                           dropout=0, pooling=pooling,
                           enc=True, bias=bias, 
                           aal_h=None)
            self.enc_odict['block{}'.format(layer_idx)] = block
        self.enc = nn.Sequential(self.enc_odict)
        self.enc.apply(weights_init)
        print('F0Regressor Encoder: ', self.enc)
        self.state_proj_h = nn.Linear(enc_fmaps[-1] * 8, rnn_size)
        self.state_proj_c = nn.Linear(enc_fmaps[-1] * 8, rnn_size)
        self.rnn = nn.LSTM(rnn_size, rnn_size, 
                           batch_first=True)
        self.fc_lf0 = nn.Linear(rnn_size, 1)
        self.fc_uv = nn.Linear(rnn_size, 1)
        # reverse direction for feedback
        self.out_emb = nn.Linear(2, rnn_size)

    def forward(self, x, num_steps):
        h = self.enc(x)
        #print('h size: ', h.size())
        h = h.transpose(1, 2).contiguous()
        h = h.view(h.size(0), -1)
        #print('rnn input size: ', h.size())
        # initial states for RNN will be projection of encoder state
        state_h = self.state_proj_h(h).unsqueeze(0)
        state_c = self.state_proj_c(h).unsqueeze(0)
        # first step
        hT = Variable(torch.zeros(h.size(0), 1, self.rnn_size))
        if self.do_cuda:
            hT = hT.cuda()
        states = []
        outs = [None] * num_steps
        for t_ in range(num_steps):
            hT, state_t = self.rnn(hT, (state_h, state_c))
            state_h, state_c = state_t
            out_lf0 = self.fc_lf0(hT)
            out_uv = F.sigmoid(self.fc_uv(hT))
            out_h = torch.cat((out_lf0, out_uv), dim=2)
            outs[t_] = out_h
            hT = self.out_emb(out_h)
        outs = torch.cat(outs, dim=1)
        #print('out_h size: {}'.format(out_h.size()))
        return outs

class F0Regressor(Model):

    def __init__(self, opts, name='F0Regressor'):
        super().__init__(name)
        self.opts = opts
        # max_pad operates on G input to translate signal here and there
        self.save_path = opts.save_path
        self.do_cuda = opts.cuda
        self.pooling_size=opts.pooling_size
        self.f0_evaluator = F0Evaluator(cuda=opts.cuda)
        kwidth = opts.kwidth
        lnorm = opts.lnorm
        self.enc_odict = OrderedDict()
        enc_fmaps = opts.enc_fmaps
        enc_acts = [nn.PReLU(fmaps) for fmaps in enc_fmaps]
        self.f0reg = F0Predictor(enc_fmaps, enc_acts, opts.kwidth,
                                 pooling=opts.pooling_size, lnorm=lnorm,
                                 rnn_size=opts.rnn_size,
                                 cuda=opts.cuda)
        # create writer
        self.writer = SummaryWriter(os.path.join(opts.save_path, 'train'))
        self.saver = Saver(self, opts.save_path, max_ckpts=120)


    def train_(self, opts, dloader, criterion, 
               log_freq, va_dloader=None, smooth=0):

        """ Train the SEGAN """
        opt = optim.Adam(self.f0reg.parameters(), opts.lr)
        num_batches = len(dloader) 

        self.train()
        
        global_step = 1
        timings = []
        for epoch in range(1, opts.epoch + 1):
            beg_t = timeit.default_timer()
            for bidx, batch in enumerate(dloader, start=1):
                opt.zero_grad()
                sample = batch
                uttname, whisper, lf0, uv = batch
                whisper = Variable(whisper.unsqueeze(1))
                lf0 = Variable(lf0)
                uv = Variable(uv)
                if self.do_cuda:
                    whisper = whisper.cuda()
                    lf0 = lf0.cuda()
                    uv = uv.cuda()
                hts = self.f0reg(whisper, lf0.size(1))
                lab = torch.cat((lf0.unsqueeze(2), uv.unsqueeze(2)), dim=2)
                loss = criterion(hts.view(-1, hts.size(2)), 
                                 lab.view(-1, lab.size(2)))
                loss.backward()
                opt.step()

                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if bidx % log_freq == 0 or bidx >= len(dloader):
                    loss_v = np.asscalar(loss.cpu().data.numpy())
                    log = '(Iter {}) Batch {}/{} (Epoch {}) loss:{:.4f}, ' \
                          ''.format(global_step, bidx,
                                    len(dloader), epoch,
                                    loss_v)
                    log += 'btime: {:.4f} s, mbtime: {:.4f} s' \
                           ''.format(timings[-1],
                                     np.mean(timings))
                    print(log)
                    self.writer.add_scalar('Loss', loss_v, global_step)
                    lf0_his = hts[:, :, 0].contiguous().view(-1,)
                    uv_his = hts[:, :, 1].contiguous().view(-1,)
                    gt_lf0_his = lf0.contiguous().view(-1,)
                    gt_uv_his = uv.contiguous().view(-1,)
                    self.writer.add_histogram('Pred-iLF0',
                                              lf0_his.cpu().data.numpy(),
                                              global_step,
                                              bins='sturges')
                    self.writer.add_histogram('GT-iLF0',
                                              gt_lf0_his.cpu().data.numpy(),
                                              global_step,
                                              bins='sturges')
                    self.writer.add_histogram('Pred-UV', 
                                              uv_his.cpu().data.numpy(),
                                              global_step,
                                              bins='sturges')
                    self.writer.add_histogram('GT-UV', 
                                              gt_uv_his.cpu().data.numpy(),
                                              global_step,
                                              bins='sturges')
                    # get D and G weights and plot their norms by layer and
                    # global
                    def model_weights_norm(model, total_name):
                        total_W_norm = 0
                        for k, v in model.named_parameters():
                            if 'weight' in k:
                                W = v.data
                                W_norm = torch.norm(W)
                                self.writer.add_scalar('{}_Wnorm'.format(k),
                                                       W_norm,
                                                       global_step)
                                total_W_norm += W_norm
                        self.writer.add_scalar('{}_Wnorm'.format(total_name),
                                               total_W_norm,
                                               global_step)
                    model_weights_norm(self.f0reg, 'Gtotal')
                    # save model
                    self.save(self.save_path, global_step)
                global_step += 1

            if va_dloader is not None:
                #pesqs, mpesq = self.evaluate(opts, va_dloader, log_freq)
                # need to trim to 10 samples cause it is slow process
                mae, acc = self.evaluate(opts, va_dloader, criterion,
                                         epoch,
                                         max_samples=10)


    def evaluate(self, opts, dloader, criterion, epoch, max_samples=100):
        """ Evaluate D_x, G_z1 and G_z2 with validation/test data """
        self.f0reg.eval()
        beg_eval_t = timeit.default_timer()
        total_s = 0
        timings = []
        spkid = None
        # store eval results from F0Evaluator
        klds = []
        maes = []
        accs = []
        # going over dataset ONCE
        for bidx, batch in enumerate(dloader, start=1):
            sample = batch
            uttname, whisper, lf0, uv = batch
            whisper_npy = whisper.numpy()
            whisper = Variable(whisper.unsqueeze(1), volatile=True)
            lf0 = Variable(lf0, volatile=True)
            uv = Variable(uv, volatile=True)
            if self.do_cuda:
                whisper = whisper.cuda()
                lf0 = lf0.cuda()
                uv = uv.cuda()
            hts = self.f0reg(whisper, lf0.size(1))
            pred_lf0, pred_uv = torch.chunk(hts, 2, dim=2)
            pred_uv = torch.round(pred_uv).squeeze(2)
            pred_lf0 = pred_lf0.squeeze(2)
            mask = pred_uv * uv
            pred_lf0 = pred_lf0 * uv
            lf0 = lf0 * uv
            mae = compute_MAE(pred_lf0, lf0)
            acc = compute_accuracy(pred_uv, uv)
            maes.append(mae.cpu().data[0])
            accs.append(acc.cpu().data[0])
            end_eval_t = timeit.default_timer()
            timings.append(end_eval_t - beg_eval_t)
            beg_eval_t = timeit.default_timer()
            print('Eval batch {}/{} computed in {} s, mbtime: {} '
                  's'.format(bidx, len(dloader), timings[-1],
                            np.mean(timings)),
                  end='\r')
            if bidx >= max_samples:
                break
        print('')
        avg_mae = np.mean(maes)
        avg_acc = np.mean(accs)
        self.writer.add_histogram('Eval-GT_UV', uv.view(-1,).cpu().data.numpy(),
                                  epoch, bins='sturges')
        self.writer.add_histogram('Eval-PRED_UV',
                                  pred_uv.view(-1,).cpu().data.numpy(),
                                  epoch, bins='sturges')
        self.writer.add_histogram('Eval-PRED_F0',
                                  torch.exp(pred_lf0.view(-1,).cpu().data).numpy(),
                                  epoch, bins='sturges')
        self.writer.add_histogram('Eval-GT_F0',
                                  torch.exp(lf0.view(-1,).cpu().data).numpy(),
                                  epoch, bins='sturges')
        self.writer.add_scalar('meanEval-MAE_Hz', avg_mae,
                               epoch)
        self.writer.add_scalar('meanEval-ACC_norm', avg_acc,
                               epoch)
        return avg_mae, avg_acc

if __name__ == '__main__':
    f0_pred = F0Predictor([10, 10, 20, 20],
                          [nn.PReLU(10),nn.PReLU(10),nn.PReLU(20),nn.PReLU(20)],
                          11, 2)
