import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
import torch.nn.functional as F
import os
import math
import json

class Saver(object):

    def __init__(self, model, save_path, max_ckpts=5, optimizer=None, prefix=''):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, '{}checkpoints'.format(prefix)) 
        self.max_ckpts = max_ckpts
        self.optimizer = optimizer
        self.prefix = prefix

    def save(self, model_name, step, best_val=False):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        ckpt_path = self.ckpt_path
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as ckpt_f:
                # read latest checkpoints
                ckpts = json.load(ckpt_f)
        else:
            ckpts = {'latest':[], 'current':[]}

        model_path = '{}-{}.ckpt'.format(model_name, step)
        if best_val: 
            model_path = 'best_' + model_path
        model_path = '{}{}'.format(self.prefix, model_path)
        
        # get rid of oldest ckpt, with is the frst one in list
        latest = ckpts['latest']
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) > self.max_ckpts:
                    try:
                        print('Removing old ckpt {}'.format(os.path.join(save_path, 
                                                            'weights_' + todel)))
                        os.remove(os.path.join(save_path, 'weights_' + todel))
                        latest = latest[1:] 
                    except FileNotFoundError:
                        print('ERROR: ckpt is not there?')

        latest += [model_path]

        ckpts['latest'] = latest
        ckpts['current'] = model_path

        with open(ckpt_path, 'w') as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))

        st_dict = {'step':step,
                   'state_dict':self.model.state_dict()}

        if self.optimizer is not None: 
            st_dict['optimizer'] = self.optimizer.state_dict()
        # now actually save the model and its weights
        #torch.save(self.model, os.path.join(save_path, model_path))
        torch.save(st_dict, os.path.join(save_path, 
                                          'weights_' + \
                                           model_path))

    def read_latest_checkpoint(self):
        ckpt_path = self.ckpt_path
        print('Reading latest checkpoint from {}...'.format(ckpt_path))
        if not os.path.exists(ckpt_path):
            print('[!] No checkpoint found in {}'.format(self.save_path))
            return False
        else:
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts['current'] 
            return curr_ckpt

    #def load(self):
    #    save_path = self.save_path
    #    ckpt_path = self.ckpt_path
    #    print('Reading latest checkpoint from {}...'.format(ckpt_path))
    #    if not os.path.exists(ckpt_path):
    #        raise FileNotFoundError('[!] Could not load model. Ckpt '
    #                                '{} does not exist!'.format(ckpt_path))
    #    with open(ckpt_path, 'r') as ckpt_f:
    #        ckpts = json.load(ckpt_f)
    #    curr_ckpt = ckpts['curent'] 
    #    st_dict = torch.load(os.path.join(save_path, curr_ckpt))
    #    return 

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is False:
            if not os.path.exists(ckpt_path):
                print('[!] No weights to be loaded')
                return False
        else:
            st_dict = torch.load(os.path.join(save_path,
                                              'weights_' + \
                                              curr_ckpt))
            if 'state_dict' in st_dict:
                # new saving mode
                model_state = st_dict['state_dict']
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and 'optimizer' in st_dict:
                    self.optimizer.load_state_dict(st_dict['optimizer'])
            else:
                # legacy mode, only model was saved
                self.model.load_state_dict(st_dict)
            print('[*] Loaded weights')
            return True

    def load_pretrained_ckpt(self, ckpt_file, load_last=False, load_opt=True):
        model_dict = self.model.state_dict() 
        st_dict = torch.load(ckpt_file, 
                             map_location=lambda storage, loc: storage)
        if 'state_dict' in st_dict:
            pt_dict = st_dict['state_dict']
        else:
            # legacy mode
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            # Get rid of last layer params (fc output in D)
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        # Filter unnecessary keys from loaded ones and those not existing
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and \
                   k in allowed_keys and v.size() == model_dict[k].size()}
        print('Current Model keys: ', len(list(model_dict.keys())))
        print('Loading Pt Model keys: ', len(list(pt_dict.keys())))
        print('Loading matching keys: ', list(pt_dict.keys()))
        if len(pt_dict.keys()) != len(model_dict.keys()):
            print('WARNING: LOADING DIFFERENT NUM OF KEYS')
        # overwrite entries in existing dict
        model_dict.update(pt_dict)
        # load the new state dict
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                print('WARNING: {} weights not loaded from pt ckpt'.format(k))
        if self.optimizer is not None and 'optimizer' in st_dict and load_opt:
            self.optimizer.load_state_dict(st_dict['optimizer'])


class Model(nn.Module):

    def __init__(self, name='BaseModel'):
        super().__init__()
        self.name = name
        self.optim = None

    def save(self, save_path, step, best_val=False, saver=None):
        model_name = self.name

        if not hasattr(self, 'saver') and saver is None:
            self.saver = Saver(self, save_path,
                               optimizer=self.optim,
                               prefix=model_name + '-')

        if saver is None:
            self.saver.save(model_name, step, best_val=best_val)
        else:
            # save with specific saver
            saver.save(model_name, step, best_val=best_val)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, 'saver'):
                self.saver = Saver(self, save_path, 
                                   optimizer=self.optim,
                                   prefix=model_name + '-')
            self.saver.load_weights()
        else:
            print('Loading ckpt from ckpt: ', save_path)
            # consider it as ckpt to load per-se
            self.load_pretrained(save_path)

    def load_pretrained(self, ckpt_path, load_last=False):
        # tmp saver
        saver = Saver(self, '.', optimizer=self.optim)
        saver.load_pretrained_ckpt(ckpt_path, load_last)


    def activation(self, name):
        return getattr(nn, name)()

    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

class LayerNorm(nn.Module):

    def __init__(self, *args):
        super().__init__()

    def forward(self, activation):
        if len(activation.size()) == 3:
            ori_size = activation.size()
            activation = activation.view(-1, activation.size(-1))
        else:
            ori_size = None
        means = torch.mean(activation, dim=1, keepdim=True)
        stds = torch.std(activation, dim=1, keepdim=True)
        activation = (activation - means) / stds
        if ori_size is not None:
            activation = activation.view(ori_size)
        return activation



class VirtualBatchNorm1d(Module):
    """
    Module for Virtual Batch Normalization.
    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """
    def __init__(self, num_features, eps=1e-5, cuda=False):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        self.ref_mean = self.register_parameter('ref_mean', None)
        self.ref_mean_sq = self.register_parameter('ref_mean_sq', None)

        # define gamma and beta parameters
        gamma = torch.normal(means=torch.ones(1, num_features, 1), std=0.02)
        beta = torch.FloatTensor(1, num_features, 1).fill_(0)
        if cuda:
            gamma = gamma.cuda()
            beta = beta.cuda()

        self.gamma = Parameter(gamma.float())
        self.beta = Parameter(beta)

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean: None, ref_mean_sq: None):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.
        The input parameter is_reference should indicate whether it is a forward pass
        for reference batch or not.
        Args:
            x: input tensor
            is_reference(bool): True if forwarding for reference batch
        Result:
            x: normalized batch tensor
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self._normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self._normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def _normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.
        Args:
            x: input tensor
            mean: mean over features. it has size [1:num_features:]
            mean_sq: squared means over features.
        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception(
                    'Mean size not equal to number of featuers : given {}, expected {}'
                    .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception(
                    'Squared mean tensor size not equal to number of features : given {}, expected {}'
                    .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
				.format(name=self.__class__.__name__, **self.__dict__))


class Conv1DResBlock(nn.Module):

    def __init__(self, ninputs, fmaps, kwidth=3, 
                 dilations=[1, 2, 4, 8], stride=4, bias=True, 
                 transpose=False, act='prelu'):
        super().__init__()
        self.ninputs = ninputs
        self.fmaps = fmaps
        self.kwidth = kwidth
        self.dilations = dilations
        self.stride = stride
        self.bias = bias
        self.transpose = transpose
        assert dilations[0] == 1, dilations[0]
        assert len(dilations) > 1, len(dilations)
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        prev_in = ninputs
        for n, d in enumerate(dilations):
            if n == 0:
                curr_stride = stride
            else:
                curr_stride = 1
            if n == 0 or (n + 1) >= len(dilations):
                # in the interfaces in/out it is different
                curr_fmaps = fmaps
            else:
                curr_fmaps = fmaps // 4
                # minimum is 1
                curr_fmaps = max(curr_fmaps, 1)
            if n == 0 and transpose:
                p_ = (self.kwidth - 4)//2
                op_ = 0
                if p_ < 0:
                    op_ = p_ * -1
                    p_ = 0
                self.convs.append(nn.ConvTranspose1d(prev_in, curr_fmaps, kwidth,
                                                     stride=curr_stride,
                                                     dilation=d, 
                                                     padding=p_,
                                                     output_padding=op_,
                                                     bias=bias))
            else:
                self.convs.append(nn.Conv1d(prev_in, curr_fmaps, kwidth,
                                            stride=curr_stride,
                                            dilation=d, 
                                            padding=0,
                                            bias=bias))
            self.acts.append(nn.PReLU(curr_fmaps))
            prev_in = curr_fmaps

    def forward(self, x):
        h = x
        res_act = None
        for li, layer in enumerate(self.convs):
            if self.stride > 1 and li == 0:
                # add proper padding
                pad_tuple = ((self.kwidth//2)-1, self.kwidth//2)
            else:
                # symmetric padding
                p_ = ((self.kwidth - 1) * self.dilations[li]) // 2
                pad_tuple = (p_, p_)
            #print('Applying pad tupple: ', pad_tuple)
            if not (self.transpose and li == 0):
                h = F.pad(h, pad_tuple)
            #print('Layer {}'.format(li))
            #print('h padded: ', h.size())
            h = layer(h)
            h = self.acts[li](h)
            if li == 0:
                # keep the residual activation
                res_act = h
            #print('h min: ', h.min())
            #print('h max: ', h.max())
            #print('h conved size: ', h.size())
        # add the residual activation in the output of the module
        return h + res_act

def pos_code(chunk_pos, x):
    # positional code
    pos_dim = x.size(1)
    chunk_size = x.size(2)
    bsz = x.size(0)
    pe = torch.zeros(x.size(0), chunk_size, pos_dim)
    for n in range(bsz):
        cpos = chunk_pos[n].item()
        position = torch.arange(chunk_size * cpos,
                                chunk_size * cpos + chunk_size)
        position = position.unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2) *
            -(math.log(10000.0) / pos_dim))
        pe[n, :, 0::2] = torch.sin(position * div_term)
        pe[n, :, 1::2] = torch.cos(position * div_term)
    pe = pe.transpose(1, 2)
    if x.is_cuda:
        pe = pe.to('cuda')
    x = x + pe
    return x

