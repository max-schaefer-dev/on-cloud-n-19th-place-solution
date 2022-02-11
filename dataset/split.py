import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
    
# DATA SPLIT
def create_folds(df, CFG=None):
    df = df.copy()

    selected_fold = CFG.selected_folds

    feature_cols = ["chip_id"] + [f"{band}_path" for band in CFG.selected_bands]

    val = df.loc[df['fold'] == selected_fold].reset_index(drop=True)
    val_X = val[feature_cols].copy()
    val_y = val[["chip_id", "label_path"]].copy()

    train = df.loc[df['fold'] != selected_fold].reset_index(drop=True)
    train_X = train[feature_cols].copy()
    train_y = train[["chip_id", "label_path"]].copy()

    print(f'selected fold {selected_fold}: {len(train_X)} train / {len(val_X)} val. split. {round(len(val_X)/len(df),2)}%')

    return (train_X, train_y, val_X, val_y)