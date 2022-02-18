import zipfile
from pathlib import Path

def prepare_data(data_dir):
    """ Unzips the pseudo labels needed for training
    Args:
        data_dir (Path): path to the dataset directory
    """
    if type(data_dir) == Path:
        path_to_zip_file = data_dir / 'pseudo_labels.zip'
    else:
        path_to_zip_file = data_dir + '/pseudo_labels.zip'

    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)