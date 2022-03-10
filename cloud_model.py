from typing import Optional, List

import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics
import torch.nn.functional as F

from utils.schedulers import get_lr_scheduler
from utils.optimizers import get_optimizer
from utils.losses import get_loss
from utils.metrics import JaccardIndex

from dataset.cloud_dataset import CloudDataset

class CloudModel(pl.LightningModule):
    def __init__(
        self,
        bands: List[str],
        x_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        x_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        hparams: dict = {},
    ):
        """
        Instantiate the CloudModel class based on the pl.LightningModule
        (https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).

        Args:
            bands (list[str]): Names of the bands provided for each chip
            x_train (pd.DataFrame, optional): a dataframe of the training features with a row for each chip.
                There must be a column for chip_id, and a column with the path to the TIF for each of bands.
                Required for model training
            y_train (pd.DataFrame, optional): a dataframe of the training labels with a for each chip
                and columns for chip_id and the path to the label TIF with ground truth cloud cover.
                Required for model training
            x_val (pd.DataFrame, optional): a dataframe of the validation features with a row for each chip.
                There must be a column for chip_id, and a column with the path to the TIF for each of bands.
                Required for model training
            y_val (pd.DataFrame, optional): a dataframe of the validation labels with a for each chip
                and columns for chip_id and the path to the label TIF with ground truth cloud cover.
                Required for model training
            hparams (dict, optional): Dictionary of additional modeling parameters.
        """
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        # required
        self.bands = bands

        # optional modeling params
        self.encoder = self.hparams.get('encoder', 'resnet34') #resnet34
        self.decoder = self.hparams.get('decoder', 'unet')
        self.weights = self.hparams.get('weights', 'imagenet')
        self.learning_rate = self.hparams.get('lr', 1e-3)
        self.patience = self.hparams.get('patience', 4)
        self.num_workers = self.hparams.get('num_workers', 2)
        self.batch_size = self.hparams.get('batch_size', 16)
        self.gpu = self.hparams.get('gpu', False)
        self.classes = self.hparams.get('classes', 4)
        self.train_transform = self.hparams.get('train_transform', None)
        self.val_transform = self.hparams.get('val_transform', None)

        # Instantiate datasets, model, and trainer params if provided
        self.train_dataset = CloudDataset(
            x_paths=x_train,
            y_paths=y_train,
            bands=self.bands,
            transforms=self.train_transform,
        )

        self.val_dataset = CloudDataset(
            x_paths=x_val,
            y_paths=y_val,
            bands=self.bands,
            transforms=self.val_transform,
        )
        self.model = self._prepare_model()

    ## Required LightningModule methods ##

    def forward(self, image: torch.Tensor):
        # Forward pass
        embedding = self.model(image)
        return embedding

    def training_step(self, batch: dict, batch_idx: int):
        """
        Training step.

        Args:
            batch (dict): dictionary of items from CloudDataset of the form
                {'chip_id': list[str], 'chip': list[torch.Tensor], 'label': list[torch.Tensor]}
            batch_idx (int): batch number
        """
        if self.train_dataset.data is None:
            raise ValueError(
                "x_train and y_train must be specified when CloudModel is instantiated to run training"
            )

        # Switch on training mode
        self.model.train()
        torch.set_grad_enabled(True)

        # Load images and labels
        x = batch["chip"]

        y = batch["label"].long()

        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass
        preds = self.forward(x)

        # Log batch loss
        loss = get_loss(self.hparams.loss)
        loss = loss(preds, y).mean()

        self.log(
            "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.batch_size,
        )

        return loss
    

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Validation step.

        Args:
            batch (dict): dictionary of items from CloudDataset of the form
                {'chip_id': list[str], 'chip': list[torch.Tensor], 'label': list[torch.Tensor]}
            batch_idx (int): batch number
        """
        if self.val_dataset.data is None:
            raise ValueError(
                "x_val and y_val must be specified when CloudModel is instantiated to run validation"
            )

        # Switch on validation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # Load images and labels
        x = batch["chip"]
        y = batch["label"] #.long()
        if self.gpu:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Forward pass & softmax
        preds = self.forward(x)

        # Log batch loss
        loss = get_loss(self.hparams.loss)
        loss = loss(preds, y.long()).mean()
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        # Forward pass & softmax
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5) * 1  # convert to int

        # Log batch IOU
        jaccardIndex = JaccardIndex(preds, y)

        self.log(
            "jaccardIndex",
            jaccardIndex,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

        return jaccardIndex


    def train_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        opt = get_optimizer(self.model.parameters(),
                            self.hparams.lr,
                            self.hparams.optimizer
                            )
        sch = get_lr_scheduler(opt, self.hparams.scheduler)
        return [opt], [sch]

    ## Convenience Methods ##

    def _prepare_model(self):
        # Instantiate U-Net model
        aux_params = {
        }
        if self.decoder == 'unet':
            unet_model = smp.Unet(
                encoder_name=self.encoder,
                encoder_weights=self.weights,
                in_channels=4,
                classes=self.classes
            )
        elif self.decoder == 'fpn':
            unet_model = smp.FPN(
                encoder_name=self.encoder,
                encoder_weights=self.weights,
                in_channels=4,
                classes=self.classes
            )
        else:
            raise ValueError("Wrong decoder name")

        if self.gpu:
            unet_model.cuda()

        return unet_model