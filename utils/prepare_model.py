from cloud_model import CloudModel
from dataset.split import create_folds
from utils.config import cfg2dict

def prepare_model(CFG, df):

    cfg_dict = cfg2dict(CFG)

    if not CFG.all_data:
        train_X, train_y, val_X, val_y = create_folds(df, CFG=CFG)
    else:
        train_X, train_y = create_folds(df, CFG=CFG)

    if not CFG.all_data:
        cloud_model = CloudModel(
            bands=CFG.selected_bands,
            x_train=train_X,
            y_train=train_y,
            x_val=val_X,
            y_val=val_y,
            hparams=cfg_dict
        )
    else:
        cloud_model = CloudModel(
            bands=CFG.selected_bands,
            x_train=train_X,
            y_train=train_y,
            hparams=cfg_dict
        )

    return cloud_model