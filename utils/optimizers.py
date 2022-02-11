import torch

def get_optimizer(model_params, lr, opt_name):
    if opt_name=='Adam':
        opt = torch.optim.Adam(params=model_params, lr=lr)
    elif opt_name=='AdamW':
        opt = torch.optim.AdamW(params=model_params, lr=lr)
    elif opt_name=='RAdam':
        opt = torch.optim.RAdam(params=model_params, lr=lr)
    else:
        raise ValueError("Wrong optimizer name")
    return opt