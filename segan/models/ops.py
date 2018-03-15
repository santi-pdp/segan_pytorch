import torch.optim as optim
from torch.optim import lr_scheduler


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
