import pandas as pd
import os
import glob
import shutil
from tqdm.notebook import tqdm

def prepare_data(data_dir):

    TEST_DIRECTORY = f'{data_dir}/test_features'

    if not os.path.isdir(TEST_DIRECTORY):
        os.makedirs(TEST_DIRECTORY)

    train_f_paths = glob.glob(f'{data_dir}/train_features/*')

    for p in tqdm(train_f_paths[:1000]):
        f_name = os.path.split(p)[1]
        print('trigger')
        if not os.path.isdir(TEST_DIRECTORY + '/' + f_name):
            os.makedirs(TEST_DIRECTORY + '/' + f_name)
            shutil.copytree(p, TEST_DIRECTORY + '/' + f_name)


def update_filepaths(df, bands, ds_path):
    """Updates the image paths to the correct data directory provided by CFG.data

    Args:
        df (pd.DataFrame): full dataframe
        CFG: python class object as config
        
    Returns:
        updated_df (pd.DataFrame): dataframe with updated filepaths.
    """
    updated_df = df.copy(deep=True)
    
    for band in bands:
        updated_df[f'{band}_path'] = str(ds_path) + '/train_features/' + updated_df.chip_id + f'/{band}.tif'
        # updated_df[f'{band}_path'] = str(ds_path) + f'/train_features/{updated_df.chip_id}/{band}.tif'

    # updated_df['label_path'] = ds_path / f'/train_labels/{updated_df.chip_id}.tif'
    updated_df['label_path'] = str(ds_path) + '/train_labels/' + updated_df.chip_id + '.tif'
    return updated_df