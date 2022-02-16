import os
from pathlib import Path
from typing import List

from loguru import logger
import pandas as pd
from PIL import Image
import torch
import typer
import argparse
import yaml
import glob

from cloud_model import CloudModel
from utils.config import dict2cfg
from dataset.augment import prepare_train_augmentation, prepare_val_augmentation
from dataset.cloud_dataset import CloudDataset
from dataset.processing import update_filepaths, prepare_data
from dataset.split import create_folds
from utils.get_metadata import get_metadata
from utils.prepare_tta import prepare_tta

DATA_DIRECTORY = Path("./data")
PREDICTIONS_DIRECTORY = DATA_DIRECTORY / "predictions"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"
BANDS = ["B02", "B03", "B04", "B08"]

# Set the pytorch cache directory and include cached models in your submission.zip
# os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "assets/torch")

def predict(
    # model_weights_path: Path = ASSETS_DIRECTORY / "cloud_model.pt",
    CFG,
    model_paths: list = [],
    config_paths: list = [],
    batch_size: int = 8,
    num_workers: int = 2,
    test_features_dir: Path = DATA_DIRECTORY / "test_features",
    predictions_dir: Path = PREDICTIONS_DIRECTORY,
    fast_dev_run: bool = False,
):
    """
    Generate predictions for the chips in test_features_dir using the model saved at
    model_weights_path.

    Predictions are saved in predictions_dir. The default paths to all three files are based on
    the structure of the code execution runtime.

    Args:
        model_weights_path (os.PathLike): Path to the weights of a trained CloudModel.
        test_features_dir (os.PathLike, optional): Path to the features for the test data. Defaults
            to 'data/test_features' in the same directory as main.py
        predictions_dir (os.PathLike, optional): Destination directory to save the predicted TIF masks
            Defaults to 'predictions' in the same directory as main.py
        bands (List[str], optional): List of bands provided for each chip
    """
    if not test_features_dir.exists():
        raise ValueError(
            f"The directory for test feature images must exist and {test_features_dir} does not exist"
        )
    predictions_dir.mkdir(exist_ok=True, parents=True)

    # META DATA
    df = pd.read_csv(F'./data/train_metadata_cleaned_kfold.csv')
    df = update_filepaths(df, BANDS, DATA_DIRECTORY)

    # test_features_dir = Path('/content/on-cloud-n-19th-place-solution/data/test_features')
    # PREPARE DATA
    logger.info("Loading test metadata")
    test_metadata = get_metadata(test_features_dir, bands=BANDS)
    if CFG.fast_dev_run:
        test_metadata = test_metadata.head(50)
    logger.info(f"Found {len(test_metadata)} chips")

    test_dataset = CloudDataset(x_paths=test_metadata, bands=BANDS)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    models = {}

    # PREPARE MODELS:
    for model_path, config_path in zip(model_paths, config_paths):

        # LOADING CONFIG
        cfg_dict  = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        MODEL_CFG = dict2cfg(cfg_dict) # dict to class

        # PREPARE AUGMENTATIONS
        cfg_dict['train_transform'] = prepare_train_augmentation()
        cfg_dict['val_transform'] = prepare_val_augmentation()

        cloud_model = CloudModel(bands=BANDS, hparams=cfg_dict)
        cloud_model.load_state_dict(torch.load(model_path))
        cloud_model.eval()

        cloud_model = prepare_tta(cloud_model, CFG)

        models[MODEL_CFG.model_name] = cloud_model

    for batch_index, batch in enumerate(test_dataloader):
        logger.debug(f"Predicting batch {batch_index} of {len(test_dataloader)}")
        x = batch["chip"].to('cuda')

        if CFG.ensemble:
            all_preds = []
            for model_name, model in models.items():
                preds = model.forward(x)
                all_preds.append(preds)

                preds = torch.stack(all_preds)
                preds = torch.mean(preds, axis=0)
        else:
            model_name, model = list(models.items())[0]
            preds = model.forward(x)


        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5).detach().to('cpu').numpy().astype("uint8")
        for chip_id, pred in zip(batch["chip_id"], preds):
            chip_pred_path = predictions_dir / f"{chip_id}.tif"
            chip_pred_im = Image.fromarray(pred)
            chip_pred_im.save(chip_pred_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='model', help='where checkpoint weights can be found')
    parser.add_argument('--batch-size', type=int, default=8, help='number of TTA')
    parser.add_argument('--ensemble', type=int, default=1, help='use ensemble mode')
    parser.add_argument('--fast-dev-run', type=int, default=0, help='process only small portion in debug mode')
    parser.add_argument('--output-dir', type=str, default='submission', help='output path to save the submission')
    parser.add_argument('--tta', type=int, default=1, help='number of TTA')
    CFG = parser.parse_args()

    if CFG.ensemble:
        model_paths = glob.glob(f'{CFG.model_dir}/*/*.pt')
        config_paths = glob.glob(f'{CFG.model_dir}/*/*.yaml')
    else:
        model_paths = glob.glob(f'{CFG.model_dir}/*.pt')
        config_paths = glob.glob(f'{CFG.model_dir}/*.yaml')

    # Prepare data
    # prepare_data(DATA_DIRECTORY)

    # Inference
    predict(
        CFG=CFG,
        model_paths = model_paths,
        config_paths = config_paths,
        batch_size = CFG.batch_size
    )

    logger.info(f"""Saved {len(list(PREDICTIONS_DIRECTORY.glob("*.tif")))} predictions""")