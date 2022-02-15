import torch

def get_lr_scheduler(opt, scheduler_name):
    """Returns the learning rate scheduler out of the pytorch library.
    
    Args:
        CFG: python class object as config
        
    Returns:
        scheduler (torch.optim.lr_scheduler.*): learning rate scheduler out of the pytorch library.
    """
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    else:
        raise NotImplemented

    return scheduler