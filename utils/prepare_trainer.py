import pytorch_lightning as pl
from utils.callbacks import get_callbacks

def prepare_trainer(CFG):
    """
    Creates a pl.Trainer object with provided CFG.

    Args:
        CFG: python class object as config
    Returns:
        iou (pl.Trainer): pytorch lightning Trainer
    """

    limit_val_batches = 0 if CFG.all_data else 1.0
    log_every_n_steps = 1 if CFG.fast_dev_run else 50
        
    trainer = pl.Trainer(
        gpus=1,
        fast_dev_run=CFG.fast_dev_run,
        callbacks=get_callbacks(CFG),
        max_epochs=CFG.epochs,
        limit_val_batches=limit_val_batches,
        default_root_dir=CFG.output_dir,
        log_every_n_steps=log_every_n_steps
    )

    return trainer