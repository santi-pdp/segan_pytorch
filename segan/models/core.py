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

        model_path = model_name + '.ckpt'
        if best_val: 
            model_path = 'best_' + model_path
        
        # get rid of oldest ckpt, with is the frst one in list
        latest = ckpts['latest']
        if len(latest) > 0:
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
        ckpt_path = self.ckpt_path
        print('Reading latest checkpoint from {}...'.format(ckpt_path))
        if not os.path.exists(ckpt_path):
            print('[!] No weights to be loaded')
            return False
        else:
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts['current'] 
            self.model.load_state_dict(torch.load(os.path.join(save_path,
                                                               'weights_' + \
                                                               curr_ckpt)))
            print('[*] Loaded weights')
            return True

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
        if not hasattr(self, 'saver'):
            self.saver = Saver(self, save_path)
        self.saver.load_weights()

    def activation(self, name):
        return getattr(nn, name)()
