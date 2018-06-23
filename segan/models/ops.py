import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from ahoproc_tools.io import *
from ahoproc_tools.interpolate import *
import multiprocessing as mp
from scipy.io import wavfile
import tempfile
import timeit
import glob
import os

import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter


def get_grads(model):
    grads = None
    for i, (k, param) in enumerate(dict(model.named_parameters()).items()):
        if param.grad is None:
            print('WARNING getting grads: {} param grad is None'.format(k))
            continue
        if grads is None:
            grads = param.grad.cpu().data.view((-1, ))
        else:
            grads = torch.cat((grads, param.grad.cpu().data.view((-1,))), dim=0)
    return grads

def make_optimizer(otype, params, lr, step_lr=None, lr_gammma=None,
                   adam_beta1=0.7, weight_decay=0.):
    if otype == 'rmsprop':
        opt = optim.RMSprop(params, lr=lr, 
                            weight_decay=weight_decay)
    else:
        opt = optim.Adam(params, lr=lr,
                         betas=(adam_beta1, 0.9),
                         weight_decay=weight_decay)
    if step_lr is not None:
        sched = lr_scheduler.StepLR(opt, step_lr, lr_gamma)
    else:
        sched = None
    return opt, sched


def KLD(mean_p, std_p, mean_g, std_g):
    # assumping 2 normal distributions with respective mean and stds
    # log(var_g / var_p) + (var_p + (mean_p - mean_g)^2)/( 2*var_g) - 0.5
    var_p = std_p ** 2
    var_g = std_g ** 2
    num = var_p + (mean_p - mean_g) ** 2
    #print('mean_g: ', mean_g)
    #print('mean_p: ', mean_p)
    #print('std_g: ', std_g)
    #print('std_p: ', std_p)
    #print('var_g: ', var_g)
    #print('var_p: ', var_p)
    return torch.log(std_g / std_p + 1e-22) +  (num / (2 * var_g + 1e-22)) - 0.5 

def compute_MAE(v_lf0, v_ref_lf0, mask):
    #if len(v_lf0.size()) == 2:
    #    v_lf0 = v_lf0.view(-1)
    #    v_ref_lf0 = v_ref_lf0.view(-1)
    #return torch.mean(torch.abs(torch.exp(v_lf0) - torch.exp(v_ref_lf0)))
    print(mask.size())
    print(v_lf0.size())
    print(v_ref_lf0.size())
    if mask.size(1) > v_lf0.size(1):
        mask = mask[:, :v_lf0.size(1)]
        v_ref_lf0 = v_ref_lf0[:, :v_lf0.size(1)]
    if mask.size(1) < v_lf0.size(1):
        v_lf0 = v_lf0[:, :mask.size(1)]
    abs_dif = torch.abs(torch.exp(v_lf0) - torch.exp(v_ref_lf0)) * mask
    return torch.sum(abs_dif, dim=1) / torch.sum(mask, dim=1)

def compute_accuracy(uv, ref_uv):
    if ref_uv.size(1) > uv.size(1):
        ref_uv = ref_uv[:, :uv.size(1)]
    return torch.mean(uv.eq(ref_uv.view_as(uv)).float().cpu(), dim=1)

def convert_wav(wav):
    # denorm and store wav to tmp file
    f = tempfile.NamedTemporaryFile(delete=False)
    fname = f.name
    ii16 = np.iinfo(np.int16)
    wav = wav * ii16.min
    wav = wav.astype(np.int16)
    wavfile.write(f, 16000, wav)
    #print('stored wav in file: ', fname)
    # convert gwav to aco feats
    aco_name = wav2aco(fname)
    if os.path.exists(aco_name + '.lf0'):
        lf0 = read_aco_file(aco_name + '.lf0', (-1, 1))
        ilf0, uv =  interpolation(lf0, -10000000000)
        return ilf0, uv, fname
    else:
        # ahocoder can be random
        return None, None, None

def select_voiced(params):
    lf0, uv, ref_lf0, ref_uv  = params
    # first mask out unvoiced values from uvs
    mask = uv * ref_uv
    if np.sum(mask) == 0:
        return [], []
    v_lf0 = lf0[np.where(mask > 0)]
    v_ref_lf0 = ref_lf0[np.where(mask > 0)]
    return v_lf0, v_ref_lf0

class F0Evaluator(object):

    def __init__(self, f0_gtruth_dir=None,  num_proc=30, cuda=False):
        self.f0_gtruth_dir = f0_gtruth_dir
        self.pool = mp.Pool(num_proc)
        self.cuda = cuda
        # load lf0 groundtruth curves from f0_gtruth_dir
        if f0_gtruth_dir is not None:
            raise NotImplementedError
            # TODO: finish calling these dicts
            self.utt2lf0 = {}
            self.utt2uv = {}
            lf0_fnames = glob.glob(os.path.join(f0_gtruth_dir, '*.lf0'))
            for l_i, lf0_fname in enumerate(lf0_fnames, start=1):
                print('Loading {}/{} lf0 file from {}...'.format(l_i,
                                                                 len(lf0_fnames),
                                                                 self.f0_gtruth_dir),
                     end='\r')
                bname = os.path.splitext(os.path.basename(lf0_fname))[0]
                glf0 = read_aco_file(lf0_fname, (-1, 1))
                ilf0, uv = interpolation(glf0, -10000000000)
                self.utt2lf0[bname] = ilf0
                self.utt2uv[bname] = uv
            print('')


    def compute_KLD(self, v_lf0, v_ref_lf0, mask):
        #if len(v_lf0.size()) == 2:
        #    v_lf0 = v_lf0.view(-1)
        #    v_ref_lf0 = v_ref_lf0.view(-1)
        print('mask size: ', mask.size())
        #print('seq_mask size: ', seq_mask.size())
        means_p = []
        stds_p = []
        means_g = []
        stds_g = []
        for n in range(v_lf0.size(0)):
            v_n_lf0 = v_lf0[n]
            #mask_ = mask[n][:v_n_lf0.size(0)]
            v_ref_n_lf0 = v_ref_lf0[n]
            means_p.append(torch.mean(v_n_lf0))#[mask_ > 0]))
            stds_p.append(torch.std(v_n_lf0))#[mask_ > 0]))
            means_g.append(torch.mean(v_ref_n_lf0))#[mask_ > 0]))
            stds_g.append(torch.std(v_ref_n_lf0))#[mask_ > 0]))
        mean_p = torch.FloatTensor(means_p)
        std_p = torch.FloatTensor(stds_p)
        mean_g = torch.FloatTensor(means_g)
        std_g = torch.FloatTensor(stds_g)
        #mean_p = torch.mean(v_lf0 * mask * seq_mask, dim=1)
        #std_p = torch.std(v_lf0 * mask * seq_mask, dim=1)
        #mean_g = torch.mean(v_ref_lf0 * mask * seq_mask, dim=1)
        #std_g = torch.std(v_ref_lf0 * mask * seq_maskk, dim=1)
        return KLD(mean_p, std_p, mean_g, std_g), (std_p, std_g)

    def aco_eval(self, lf0_path, ref_lf0_path):
        lf0 = read_aco_file(lf0_path)
        ref_lf0 = read_aco_file(ref_lf0_path)
        lf0 = lf0[:ref_lf0.shape[0]]
        lf0, uv =  interpolation(lf0, -10000000000)
        ref_lf0, ref_uv =  interpolation(ref_lf0, -10000000000)
        lf0 = torch.FloatTensor(lf0).unsqueeze(0)
        uv = torch.FloatTensor(uv.astype(np.float32)).unsqueeze(0)
        ref_lf0 = torch.FloatTensor(ref_lf0).unsqueeze(0)
        ref_uv = torch.FloatTensor(ref_uv.astype(np.float32)).unsqueeze(0)
        mask = ref_uv
        kld, stds = self.compute_KLD(lf0, ref_lf0, mask)
        mae = compute_MAE(lf0, ref_lf0, mask)
        acc = compute_accuracy(uv, ref_uv)
        p_std = stds[0]
        kld = kld[p_std > 0]
        total_kld = kld
        total_mae = mae
        total_acc = acc
        return total_kld, total_mae, total_acc, torch.mean(mask, dim=1)

    def __call__(self, wavs, ref_wavs=None, seqlens=None):
        # TODO: atm ref_wavs MUST be specified
        assert ref_wavs is not None
        # ref_wavs: can be preloaded through f0 gruth dir or
        # computed on the fly passing them here
        # wavs: numpy array of wavs of size [batch, wav_len]
        assert len(wavs.shape) == 2, len(wavs.shape)
        if ref_wavs is not None:
            assert wavs.shape == ref_wavs.shape, ref_wavs.shape
        num_wavs = wavs.shape[0]
        beg_t = timeit.default_timer()
        results = self.pool.map(convert_wav, wavs)
        ref_results = self.pool.map(convert_wav, ref_wavs)
        end_t = timeit.default_timer()
        uvs = []
        ref_uvs = []
        ilf0s = []
        ref_ilf0s = []
        conversion_args = []
        for bidx in range(num_wavs):
            ilf0, uv, fname = results[bidx]
            ref_ilf0, ref_uv, \
            ref_fname = ref_results[bidx]
            if fname is None or ref_fname is None:
                continue
            # remove tmp files
            os.remove(ref_fname)
            os.remove(ref_fname + '.fv')
            os.remove(ref_fname + '.lf0')
            os.remove(ref_fname + '.cc')
            os.remove(fname)
            os.remove(fname + '.fv')
            os.remove(fname + '.lf0')
            os.remove(fname + '.cc')
            ref_uvs.append(ref_uv.tolist())
            uvs.append(uv.tolist())
            ref_ilf0s.append(ref_ilf0.tolist())
            ilf0s.append(ilf0.tolist())
        uvs = torch.FloatTensor(uvs).squeeze(-1)
        ref_uvs = torch.FloatTensor(ref_uvs).squeeze(-1)
        ilf0s = torch.FloatTensor(ilf0s).squeeze(-1)
        ref_ilf0s = torch.FloatTensor(ref_ilf0s).squeeze(-1)
        if self.cuda:
            uvs = uvs.cuda()
            ref_uvs = ref_uvs.cuda()
            ilf0s = ilf0s.cuda()
            ref_ilf0s = ref_ilf0s.cuda()
        #mask = uvs * ref_uvs
        mask = ref_uvs
        seq_mask = None
        if seqlens is not None:
            seq_mask = []
            for s_i, slen in enumerate(seqlens):
                curr_slen = ilf0s.size(-1)
                diff_slen = curr_slen - slen
                seq_mask.append([1] * slen + [0] * diff_slen)
            seq_mask = torch.FloatTensor(seq_mask)
        #voiced_chunks = self.pool.map(select_voiced, conversion_args)
        kld, stds = self.compute_KLD(ilf0s, ref_ilf0s, mask)#, seq_mask)
        mae = compute_MAE(ilf0s, ref_ilf0s, mask)#, seq_mask)
        acc = compute_accuracy(uvs, ref_uvs)#, seq_mask)
        # filter kld with std vals of 0 predicted
        p_std = stds[0]
        kld = kld[p_std > 0]
        total_kld = kld
        total_mae = mae
        total_acc = acc
        #total_kld = kld.mean()
        #total_mae = mae.mean()
        #total_acc = acc.mean()
        return total_kld, total_mae, total_acc


