from dataset.cloud_dataset import CloudDataset
from dataset.augment import prepare_train_augmentation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xarray
import xrspatial.multispectral as ms
import torch

def get_xarray(filepath):
    """Put images in xarray.DataArray format"""
    im_arr = np.array(Image.open(filepath))#, dtype='float32')
    return xarray.DataArray(im_arr, dims=["y", "x"])


def true_color_img(chip_id, data_dir='data/train_features'):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    chip_dir = f'{data_dir}/{chip_id}'
    red = get_xarray(f'{chip_dir}/B04.tif')
    green = get_xarray(f'{chip_dir}/B03.tif')
    blue = get_xarray(f'{chip_dir}/B02.tif')

    return ms.true_color(r=red, g=green, b=blue)

def save_batch_as_jpg(CFG, train_X, train_y, num_images):
    """Saves a plot of a sample batch of images as a .jpg in the model folder"""
    transforms = prepare_train_augmentation()

    train_dataset = CloudDataset(
            x_paths=train_X,
            y_paths=train_y,
            bands=['B02','B03','B04','B08'],
            transforms=transforms,
        )

    num_rows = num_images/5

    assert (num_rows).is_integer(), 'num_images must be devisible through 5'

    fig, axs = plt.subplots(int(num_rows),5, figsize=(25, 10), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .25, wspace=.10)

    axs = axs.ravel()

    chip_ids = iter(train_dataset)

    for i in range(num_images):
        chip_id = next(chip_ids)['chip_id']
        image = true_color_img(chip_id)

        axs[i].imshow(image)
        axs[i].set_title(chip_id)

    plt.savefig(f'{CFG.output_dir}/sample_batch.jpg',dpi=100,bbox_inches='tight',pad_inches=0)

def save_lr_scheduler_as_jpg(epochs, output_dir):
    """Saves a plot of the used lr_scheduler as a .jpg in the model folder"""

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    lrs = []

    for i in range(epochs):
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    plt.plot(lrs)
    plt.xlabel('epoch'); plt.ylabel('learnig rate')
    plt.title('Learning Rate Scheduler')
    plt.savefig(f'{output_dir}/lr_schedule.jpg',dpi=100,bbox_inches='tight',pad_inches=0)