import pytorch_lightning as pl
from utils.callbacks import get_callbacks

def prepare_trainer(CFG):
    # Set up pytorch_lightning.Trainer object
    
    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=CFG.dev_run,
        callbacks=get_callbacks(),
        max_epochs=CFG.epochs
    )

    return trainer