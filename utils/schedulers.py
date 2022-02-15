import torch

def get_lr_scheduler(opt, scheduler_name):

    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    else:
        raise NotImplemented

    return scheduler