from cloud_model import CloudModel

from pytorch_lightning.utilities.seed import seed_everything

import argparse
import yaml
import os
import pandas as pd
import albumentations as A
import torch

from utils.config import dict2cfg, cfg2dict
from utils.prepare_trainer import prepare_trainer
from utils.schedulers import save_lr_scheduler_as_jpg
from dataset.processing import update_filepaths
from dataset.split import create_folds

def train(CFG):

    cfg_dict = cfg2dict(CFG)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    cfg_dict['train_transform'] = train_transform
    cfg_dict['val_transform'] = None

    cloud_model = CloudModel(
        bands=CFG.selected_bands,
        x_train=train_X,
        y_train=train_y,
        x_val=val_X,
        y_val=val_y,
        hparams=cfg_dict
    )
    trainer = prepare_trainer(CFG)

    trainer.fit(model=cloud_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/on-cloud-n.yaml', help='config file')
    parser.add_argument('--debug', type=int, default=None, help='process only small portion in debug mode')
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
    if opt.debug is not None:
        CFG.debug = opt.debug
    print('> DEBUG MODE:', bool(CFG.debug))
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
    if opt.all_data:
        CFG.all_data = opt.all_data
    if CFG.all_data:
        CFG.selected_folds = [0]
    
    
    # DS_PATH
    if opt.ds_path is not None:
        CFG.ds_path = opt.ds_path
    if not os.path.isdir(CFG.ds_path):
        raise ValueError(f'directory, <{CFG.ds_path}> not found')
    
    print('> DS_PATH:',CFG.ds_path)
    
    # META DATA
    # ## Train Data
    df = pd.read_csv(F'{CFG.ds_path}/train_metadata_cleaned_kfold.csv')
    df = update_filepaths(df, CFG)

    # CHECK FILE FROM DS_PATH
    assert os.path.isfile(df.B02_path.iloc[0])
    print('> DS_PATH: OKAY')
    
    # DATA SPLIT
    train_X, train_y, val_X, val_y = create_folds(df, CFG=CFG)
    print(f'> SELECTED FOLD {CFG.selected_folds}: {len(train_X)} train / {len(val_X)} val. split. {round(len(val_X)/len(df),2)}%')
    
    # PLOT SOME DATA
    # fold = 0
    # fold_df = df.query('fold==@fold')[100:200]
    # paths  = fold_df.image_path.tolist()
    # labels = fold_df[CFG.target_col].values
    # ds     = build_dataset(paths, labels, cache=False, batch_size=CFG.batch_size*CFG.replicas,
    #                        repeat=True, shuffle=True, augment=True, CFG=CFG)
    # ds = ds.unbatch().batch(20)
    # batch = next(iter(ds))
    # plot_batch(batch, 5, output_dir=CFG.output_dir)
    
    # SAVE LR SCHEDULE AS JPG IN MODEL FOLDER
    save_lr_scheduler_as_jpg(CFG.epochs, CFG.output_dir)

    # Training
    print('> TRAINING:')
    # train(CFG)