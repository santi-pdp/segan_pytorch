import torch
import torch.nn as nn
import os
import json

class Saver(object):

    def __init__(self, model, save_path, max_ckpts=5):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, 'checkpoints') 
        self.max_ckpts = max_ckpts

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
        
        # get rid of oldest ckpt, with is the frst one in list
        latest = ckpts['latest']
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) > self.max_ckpts:
                    print('Removing old ckpt {}'.format(os.path.join(save_path, 
                                                        'weights_' + todel)))
                    os.remove(os.path.join(save_path, 'weights_' + todel))
                    latest = latest[1:] 
        latest += [model_path]

        ckpts['latest'] = latest
        ckpts['current'] = model_path

        with open(ckpt_path, 'w') as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))

        # now actually save the model and its weights
        #torch.save(self.model, os.path.join(save_path, model_path))
        torch.save(self.model.state_dict(), os.path.join(save_path, 
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

    def load(self):
        save_path = self.save_path
        ckpt_path = self.ckpt_path
        print('Reading latest checkpoint from {}...'.format(ckpt_path))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError('[!] Could not load model. Ckpt '
                                    '{} does not exist!'.format(ckpt_path))
        with open(ckpt_path, 'r') as ckpt_f:
            ckpts = json.load(ckpt_f)
        curr_ckpt = ckpts['curent'] 
        return torch.load(os.path.join(save_path, curr_ckpt))

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is False:
            if not os.path.exists(ckpt_path):
                print('[!] No weights to be loaded')
                return False
        else:
            self.model.load_state_dict(torch.load(os.path.join(save_path,
                                                               'weights_' + \
                                                               curr_ckpt)))
                                       #map_location=lambda storage, loc:storage)
            print('[*] Loaded weights')
            return True

    def load_pretrained_ckpt(self, ckpt_file):
        model_dict = self.model.state_dict() 
        pt_dict = torch.load(ckpt_file, 
                             map_location=lambda storage, loc: storage)
        all_pt_keys = list(pt_dict.keys())
        # Get rid of last layer params (fc output in D)
        allowed_keys = all_pt_keys[:-2]
        # Filter unnecessary keys from loaded ones
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and \
                   k in allowed_keys}
        print('Current Model keys: ', len(list(model_dict.keys())))
        print('Loading Pt Model keys: ', len(list(pt_dict.keys())))
        # overwrite entries in existing dict
        model_dict.update(pt_dict)
        # load the new state dict
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                print('WARNING: {} weights not loaded from pt ckpt'.format(k))


class Model(nn.Module):

    def __init__(self, name='BaseModel'):
        super().__init__()
        self.name = name

    def save(self, save_path, step):
        model_name = self.name

        if not hasattr(self, 'saver'):
            self.saver = Saver(self, save_path)

        self.saver.save(model_name, step)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, 'saver'):
                self.saver = Saver(self, save_path)
            self.saver.load_weights()
        else:
            print('Loading ckpt from ckpt: ', save_path)
            # consider it as ckpt to load per-se
            self.load_pretrained(save_path)

    def load_pretrained(self, ckpt_path):
        # tmp saver
        saver = Saver(self, '.')
        saver.load_pretrained_ckpt(ckpt_path)


    def activation(self, name):
        return getattr(nn, name)()

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

