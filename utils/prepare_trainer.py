import pytorch_lightning as pl
from utils.callbacks import get_callbacks

def prepare_trainer(CFG):
    # Set up pytorch_lightning.Trainer object
    
    if CFG.all_data:
        trainer = pl.Trainer(
            gpus=1,
            fast_dev_run=CFG.debug,
            callbacks=get_callbacks(CFG),
            max_epochs=CFG.epochs,
            limit_val_batches=0,
            default_root_dir=CFG.output_dir
        )
    else:
        trainer = pl.Trainer(
            gpus=1,
            fast_dev_run=CFG.debug,
            callbacks=get_callbacks(CFG),
            max_epochs=CFG.epochs,
            default_root_dir=CFG.output_dir
        )

    return trainer