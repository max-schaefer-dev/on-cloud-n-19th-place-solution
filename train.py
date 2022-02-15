from cloud_model import CloudModel

from pytorch_lightning.utilities.seed import seed_everything

import argparse
import yaml
import os
import pandas as pd
import albumentations as A
import torch

from utils.config import dict2cfg, cfg2dict
from utils.prepare_model import prepare_model
from utils.prepare_trainer import prepare_trainer
from utils.visualize import save_lr_scheduler_as_jpg, save_batch_as_jpg
from dataset.augment import prepare_train_augmentation, prepare_val_augmentation
from dataset.processing import update_filepaths
from dataset.split import create_folds

def train(CFG):

    cfg_dict = cfg2dict(CFG)

    # define augmentations
    cfg_dict['train_transform'] = prepare_train_augmentation()
    cfg_dict['val_transform'] = prepare_val_augmentation()

    cloud_model = prepare_model(CFG, df)

    trainer = prepare_trainer(CFG)
    trainer.fit(model=cloud_model)

    # save model weights after last epoch
    model_weight_path = f'{CFG.output_dir}/{CFG.model_name}.pt'
    torch.save(cloud_model.state_dict(), model_weight_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/on-cloud-n.yaml', help='config file')
    parser.add_argument('--fast-dev-run', type=int, default=None, help='process only small portion in debug mode')
    parser.add_argument('--model-name', type=str, default=None, help='name of the model')
    parser.add_argument('--img-size', type=int, nargs='+', default=None, help='image size: H x W')
    parser.add_argument('--batch-size', type=int, default=None, help='batch_size for the model')
    parser.add_argument('--loss', type=str, default=None, help='name of the loss function')
    parser.add_argument('--scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument('--selected-folds', type=int, nargs='+', default=None, help='folds to train')
    parser.add_argument('--all-data', type=int, default=None, help='use all data for training no-val')
    parser.add_argument('--ds-path', type=str, default=None, help='path to dataset')
    parser.add_argument('--output-dir', type=str, default=None, help='output path to save the model')
    opt = parser.parse_args()
    
    # LOADING CONFIG
    CFG_PATH = opt.cfg
    cfg_dict = yaml.load(open(CFG_PATH, 'r'), Loader=yaml.FullLoader)
    CFG      = dict2cfg(cfg_dict) # dict to class
    print('> CONFIG:', cfg_dict)

    # SEEDING
    seed_everything(seed=CFG.seed)

    # OVERWRITE
    if opt.model_name:
        CFG.model_name = opt.model_name
    if opt.img_size:
        assert len(opt.img_size)==2, 'image size must be H x W'
        CFG.img_size = opt.img_size
    if opt.batch_size:
        CFG.batch_size = opt.batch_size
    if opt.loss:
        CFG.loss = opt.loss
    if opt.scheduler:
        CFG.scheduler = opt.scheduler
    if opt.output_dir:
        output_dir = os.path.join(opt.output_dir, '{}-{}x{}'.format(CFG.model_name, CFG.img_size[0], CFG.img_size[1]))
        os.system(f'mkdir -p {output_dir}')
    else:
        output_dir = os.path.join('output', '{}-{}x{}'.format(CFG.model_name, CFG.img_size[0], CFG.img_size[1]))
        os.system(f'mkdir -p {output_dir}')
    CFG.output_dir = output_dir
    if opt.selected_folds:
        CFG.selected_folds = opt.selected_folds
    if opt.all_data is not None:
        CFG.all_data = opt.all_data

    if opt.fast_dev_run is not None:
        CFG.fast_dev_run = opt.fast_dev_run
    if CFG.fast_dev_run and CFG.all_data:
        CFG.all_data = 0
        print('> DEBUG MODE:', f'{bool(CFG.fast_dev_run)}. CFG.all_data set to 0',)
    else:
        print('> DEBUG MODE:', bool(CFG.fast_dev_run))

    # DS_PATH
    if opt.ds_path is not None:
        CFG.ds_path = opt.ds_path
    if not os.path.isdir(CFG.ds_path):
        raise ValueError(f'directory, <{CFG.ds_path}> not found')
    
    print('> DS_PATH:',CFG.ds_path)
    
    # META DATA
    df = pd.read_csv(F'{CFG.ds_path}/train_metadata_cleaned_kfold.csv')
    df = update_filepaths(df, CFG)

    # CHECK FILE FROM DS_PATH
    assert os.path.isfile(df.B02_path.iloc[0])
    print('> DS_PATH: OKAY')
    
    # DATA SPLIT
    if CFG.all_data:
        train_X, train_y = create_folds(df, CFG=CFG)
        print(f'> FULL DATASET IS USED FOR TRAINING: {len(train_X)} samples')
    else:
        train_X, train_y, val_X, val_y = create_folds(df, CFG=CFG)
        print(f'> SELECTED FOLD {CFG.selected_folds}: {len(train_X)} train / {len(val_X)} val. split. {round(len(val_X)/len(df),2)}%')
    
    # PLOT SOME DATA
    save_batch_as_jpg(CFG, train_X, train_y)
    
    # SAVE LR SCHEDULE AS JPG IN MODEL FOLDER
    save_lr_scheduler_as_jpg(CFG.epochs, CFG.output_dir)

    # Training
    print('> TRAINING:')
    train(CFG)