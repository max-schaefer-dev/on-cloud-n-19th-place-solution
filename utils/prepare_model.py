from cloud_model import CloudModel
from dataset.split import create_folds
from utils.config import cfg2dict

def prepare_model(CFG, df):
    """
    Creates a CloudModel object with provided CFG and dataframe.

    Args:
        CFG: python class object as config
        df (pd.DataFrame): dataframe of full dataset

    Returns:
        cloud_model (CloudModel): CloudModel object
    """

    cfg_dict = cfg2dict(CFG)

    if CFG.all_data:
        train_X, train_y = create_folds(df, CFG.bands, CFG=CFG)
        val_X, val_y = None, None
    else:
        train_X, train_y, val_X, val_y = create_folds(df, CFG.bands, CFG=CFG)
        

    cloud_model = CloudModel(
        bands=CFG.bands,
        x_train=train_X,
        y_train=train_y,
        x_val=val_X,
        y_val=val_y,
        hparams=cfg_dict
    )

    return cloud_model