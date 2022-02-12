import torchmetrics
import numpy as np

# METRIC

def JaccardIndex(pred, true):
    """
    Calculates intersection and union for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torch.Tensor): a tensor of labels

    Returns:
        iou (float): total intersection of pixels in percent
    """
    valid_pixel_mask = true.ne(255)  # valid pixel mask
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    # Intersection and union totals
    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    iou = intersection.sum() / union.sum()
    return iou

def IoULoss(pred, true):
    """
    Calculates IoU loss for a batch of images.

    Args:
        pred (torch.Tensor): a tensor of predictions
        true (torch.Tensor): a tensor of labels

    Returns:
        iou loss (float): 1 - iou
    """
    IoU = intersection_over_union(pred, true)

    return 1 - IoU