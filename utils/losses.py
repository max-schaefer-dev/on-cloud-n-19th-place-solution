import numpy as np
import torch

# LOSS
def get_loss(loss_name):
    """Splits the Dataframe into train dataset and validation dataset
    
    Args:
        loss_name (string): name of the loss used.

    Returns:
        loss: selected loss
    """
    if loss_name=='CE':
        loss = torch.nn.CrossEntropyLoss(reduction="none")
    elif loss_name=='BCE':
        loss = torch.nn.BCELoss(label_smoothing=0.01)
    elif loss_name=='Huber':
        loss = torch.nn.HuberLoss(delta=CFG.huber_delta)
    else:
        raise NotImplemented
    return loss