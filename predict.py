import os
from pathlib import Path
from typing import List

from loguru import logger
import pandas as pd
from PIL import Image
import torch
import typer

from cloud_dataset import CloudDataset
from cloud_model import CloudModel


ROOT_DIRECTORY = Path("/codeexecution")
PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"

# Set the pytorch cache directory and include cached models in your submission.zip
os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "assets/torch")

def predict(
    # model_weights_path: Path = ASSETS_DIRECTORY / "cloud_model.pt",
    models: list = [],
    test_features_dir: Path = DATA_DIRECTORY / "test_features",
    predictions_dir: Path = PREDICTIONS_DIRECTORY,
    bands: List[str] = ["B02", "B03", "B04", "B08"],
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
    df = pd.read_csv(F'{CFG.ds_path}/train_metadata_cleaned_kfold.csv')
    df = update_filepaths(df, CFG)

    # PREPARE DATA
    if CFG.all_data:
        train_X, train_y = create_folds(df, CFG=CFG)
        print(f'> FULL DATASET IS USED FOR TRAINING: {len(train_X)} samples')
    else:
        train_X, train_y, val_X, val_y = create_folds(df, CFG=CFG)
        print(f'> SELECTED FOLD {CFG.selected_folds}: {len(train_X)} train / {len(val_X)} val. split. {round(len(val_X)/len(df),2)}%')

    test_dataset = CloudDataset(x_paths=x_paths, bands=bands)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    models = {}
    # PREPARE MODELS:
    for model_path, cfg_path in zip(model_paths, cfg_paths):

        # LOADING CONFIG
        CFG_PATH = opt.cfg
        cfg_dict = yaml.load(open(CFG_PATH, 'r'), Loader=yaml.FullLoader)
        CFG      = dict2cfg(cfg_dict) # dict to class
        print('> CONFIG:', cfg_dict)

        # PREPARE AUGMENTATIONS
        cfg_dict['train_transform'] = prepare_train_augmentation()
        cfg_dict['val_transform'] = prepare_val_augmentation()

        # seg_cfg['backbone'] = 'resnet34'
        cloud_model = CloudModel(bands=bands, hparams=cfg_dict)
        model_weights_path = ASSETS_DIRECTORY / "resnet34_2y2ouvhq.pt"
        cloud_model.load_state_dict(torch.load(model_weights_path))
        cloud_model.eval()

        models[CFG.model_name] = cloud_model


    for batch_index, batch in enumerate(test_dataloader):
        logger.debug(f"Predicting batch {batch_index} of {len(test_dataloader)}")
        x = batch["chip"].to('cuda')

        all_preds = []
        for model_name, model in models.items():
            preds = model.forward(x)
            # preds = m.forward(x)
            all_preds.append(preds)

        preds = torch.stack(all_preds)
        preds = torch.mean(preds, axis=0)

        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > threshold).detach().to('cpu').numpy().astype("uint8")
        for chip_id, pred in zip(batch["chip_id"], preds):
            chip_pred_path = predictions_dir / f"{chip_id}.tif"
            chip_pred_im = Image.fromarray(pred)
            chip_pred_im.save(chip_pred_path)


    logger.info("Loading model")
    seg_model = CloudModel(bands=bands, hparams=cfg)
    seg_model.load_state_dict(torch.load(model_weights_path))
    seg_model.eval()

    logger.info("Loading test metadata")
    test_metadata = get_metadata(test_features_dir, bands=bands)
    if fast_dev_run:
        test_metadata = test_metadata.head(50)
    logger.info(f"Found {len(test_metadata)} chips")

    logger.info("Generating predictions in batches")
    make_predictions(model, test_metadata, bands, predictions_dir)

    logger.info(f"""Saved {len(list(predictions_dir.glob("*.tif")))} predictions""")