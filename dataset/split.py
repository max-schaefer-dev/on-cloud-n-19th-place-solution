import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
    
# DATA SPLIT
def create_folds(df, bands, CFG=None):
    """Splits the Dataframe into train dataset and validation dataset
    Args:
        df (pd.DataFrame): full dataframe
        CFG: python class object as config
    Returns:
        train_X, train_y, val_X, val_y (pd.DataFrame): dataframes for training and validation.
    """
    feature_cols = ["chip_id"] + [f"{band}_path" for band in bands]

    if CFG.all_data:
        train = df.copy()
        train_X = train[feature_cols].copy()
        train_y = train[["chip_id", "label_path"]].copy()

        return (train_X, train_y)
    else:
        selected_fold = CFG.selected_folds

        train = df.loc[df['fold'] != selected_fold].reset_index(drop=True)
        train_X = train[feature_cols].copy()
        train_y = train[["chip_id", "label_path"]].copy()
        
        val = df.loc[df['fold'] == selected_fold].reset_index(drop=True)
        val_X = val[feature_cols].copy()
        val_y = val[["chip_id", "label_path"]].copy()


        return (train_X, train_y, val_X, val_y)