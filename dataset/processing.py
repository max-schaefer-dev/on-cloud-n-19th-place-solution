import pandas as pd

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
        # updated_df[f'{band}_path'] = ds_path + '/train_features/' + updated_df.chip_id + f'/{band}.tif'
        updated_df[f'{band}_path'] = ds_path / f'/train_features/{updated_df.chip_id}/{band}.tif'

    updated_df['label_path'] = ds_path / f'/train_labels/{updated_df.chip_id}.tif'
    return updated_df